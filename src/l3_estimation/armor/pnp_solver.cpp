#include "l3_estimation/armor/pnp_solver.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace L3Estimation {
namespace {

// 角点深度小于该值时视为落在相机平面或相机后方。
constexpr double kMinimumCornerDepth = 1e-6;

// SP Solver::optimize_yaw 的搜索窗口宽度，单位为度。单板与双板共用同一个窗口
// 和 1 度步长，两条代价曲线才能逐点对照。
constexpr double kYawSearchRangeDegrees = 140.0;

// 枚举点数。步长恒为 1 度，所以点数与度数同值；单独取名是因为亚度细化要用它
// 在栈上开一个代价数组，那里必须是整型。
constexpr int kYawSearchSteps = 140;
static_assert(
  kYawSearchSteps == static_cast<int>(kYawSearchRangeDegrees),
  "枚举点数必须与 1 度步长下的窗口宽度一致");

// 抛物线细化的曲率下限，单位为像素。低于它说明三点近乎共线或平底，顶点无定义。
// 真正兜底的是后面的 ±0.5 格判定，这里只防除零。
constexpr double kMinimumYawCurvaturePixels = 1e-9;

// 双板配对的世界系间距范围，单位为米。相邻两板间距为 2*r*sin(π/n)：四板车是
// r*sqrt(2)，三板车是 r*sqrt(3)，配合 TrackedTarget::diverged() 认可的半径范围
// [0.1, 0.4] 给出这两个边界。低于下限只可能是同一块板被检出两次，高于上限则不是
// 同一辆车的相邻两板。
constexpr double kMinimumPairGap = 0.1;
constexpr double kMaximumPairGap = 0.75;

// 将识别类别映射为实际 PnP 几何尺寸；未知类别不参与求解。映射本身放在
// types.hpp，与 L5 火控共用同一份，避免两处各写一遍后悄悄分叉。
constexpr std::optional<ArmorType>
armorTypeFromClassId(int class_id) noexcept {
  return armorTypeOf(L2Perception::armorClassFromId(class_id));
}

// SP 的 solvePnP 与重投影全链路使用 Point3f。这里在构造时按同一顺序和同一
// float 窄化生成一次，避免 yaw 的 140 次搜索反复分配。
std::vector<cv::Point3f> armorPoints(
    double width, double height) {
  const float half_width = static_cast<float>(width / 2.0);
  const float half_height = static_cast<float>(height / 2.0);
  return {{0.0F, half_width, half_height},
          {0.0F, -half_width, half_height},
          {0.0F, -half_width, -half_height},
          {0.0F, half_width, -half_height}};
}

double spLimitRad(double angle) noexcept {
  while (angle > CV_PI) angle -= 2.0 * CV_PI;
  while (angle <= -CV_PI) angle += 2.0 * CV_PI;
  return angle;
}

// 三点抛物线细化：把 1 度整步的量化误差补回来。
//
// 整步枚举给出的是栅格上的最小点，真正的极小点几乎不会正好落在整数度上，
// 所以输出天然带一个 ±0.5 度的量化误差。而且搜索栅格锚在 barrel_yaw - 70 度、
// 跟着云台走，这个误差在帧间是 (barrel_yaw mod 1 度) 的确定性锯齿而不是白噪声，
// EKF 的白噪声 R 平均不掉它。
//
// 用胜者与左右邻居的代价过一条抛物线求顶点即可。三个代价都是扫描时已经算过的，
// 不产生任何额外的重投影——这是它值得做的全部理由，绝对量只有零点几度。
//
// 返回相对胜者的偏移，单位为格（即度）。以下情况返回 0，保持整步结果：
//   - 任一代价非有限：该采样点重投影失败；
//   - 分母接近 0：三点共线或平底，顶点无定义；
//   - 顶点跑出 ±0.5 格：抛物线只在赢下来的这一格内部有意义，越界说明局部不是
//     二次的（某个角点残差穿过零点，代价在那里有折角），此时不可信。
// 这三条合起来保证细化只在格内移动，动不了 argmin，因此引入不了整步枚举没有的
// 失效模式：最坏情况就是退回整步本身的那 ±0.5 度。
double parabolicYawOffset(double left, double center, double right) noexcept {
  if (!std::isfinite(left) || !std::isfinite(center) || !std::isfinite(right)) {
    return 0.0;
  }
  const double curvature = left - 2.0 * center + right;
  if (std::abs(curvature) < kMinimumYawCurvaturePixels) {
    return 0.0;
  }
  const double offset = 0.5 * (left - right) / curvature;
  return std::abs(offset) <= 0.5 ? offset : 0.0;
}

// 世界系 yaw 到 armor -> world 旋转。装甲板按车辆类别使用固定安装倾角，
// 只有 yaw 是自由量——这正是 yaw 搜索成立的前提。
Eigen::Matrix3d armorRotationInWorld(double yaw, ArmorName name) {
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  const double pitch = armorPitchOf(name);
  const double sin_pitch = std::sin(pitch);
  const double cos_pitch = std::cos(pitch);

  return Eigen::Matrix3d{
      {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
      {sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch},
      {-sin_pitch, 0, cos_pitch}};
}

// 一块板在一帧内的重投影常量。yaw 搜索的 140 次评估里只有 sin/cos(yaw) 会变，
// 其余全是这一帧固定的量。yaw_cost 那条路每次都要重算这些常量，还要走
// eigen2cv -> Rodrigues -> projectPoints（内部再 Rodrigues 回矩阵）一整趟
// cv::Mat 往返：sp demo 上实测单次 1.381 us，140 次就是 193 us 一块板。
//
// 这里把常量提出循环，并按 armorPoints 的结构直接算四个角点——板的局部 x 恒为
// 零，所以只需要 armor -> world 旋转的第二、三列，连矩阵都不必构造。实测单次
// 0.055 us、整块板 7.7 us，25 倍，而且不再有任何堆分配。
//
// 代价的定义（四角像素距离之和）一字未改，yaw_cost 与 reproject_armor 也原样
// 保留：后者是 public 的，auto_aim_test 和 track_diag 画代价曲线都在用。
// pnp_solver_smoke 逐点交叉核对两条路，任一条写错都会被另一条抓住。
class YawCostCache {
public:
  YawCostCache(const L1Sensor::CameraCalibration &calibration,
               const Eigen::Matrix3d &R_camera_barrel,
               const Eigen::Vector3d &t_camera_barrel,
               const Eigen::Matrix3d &R_barrel_world,
               const std::vector<cv::Point3f> &object_points,
               const Armor &armor) noexcept;

  // false 表示这份标定走不了快路，调用方必须退回 yaw_cost。
  bool usable() const noexcept { return usable_; }

  double cost(double yaw) const noexcept;

private:
  bool usable_{false};
  // world -> camera 旋转的三列。角点只用到板面的横向与竖向，这三列足够合成。
  Eigen::Vector3d axis_x_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d axis_y_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d axis_z_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d center_{Eigen::Vector3d::Zero()};
  double half_width_{0.0};
  double half_height_{0.0};
  double sin_pitch_{0.0};
  double cos_pitch_{1.0};
  double fx_{0.0};
  double fy_{0.0};
  double cx_{0.0};
  double cy_{0.0};
  double k1_{0.0};
  double k2_{0.0};
  double p1_{0.0};
  double p2_{0.0};
  double k3_{0.0};
  std::array<cv::Point2f, 4> observed_{};
};

YawCostCache::YawCostCache(const L1Sensor::CameraCalibration &calibration,
                           const Eigen::Matrix3d &R_camera_barrel,
                           const Eigen::Vector3d &t_camera_barrel,
                           const Eigen::Matrix3d &R_barrel_world,
                           const std::vector<cv::Point3f> &object_points,
                           const Armor &armor) noexcept {
  const cv::Mat &matrix = calibration.camera_matrix;
  const cv::Mat &distortion = calibration.distortion_coefficients;
  // validCalibration 允许 4/5/8/12/14 个畸变系数，这里只实现最常用的
  // k1,k2,p1,p2,k3。更长的（rational / thin-prism / tilted sensor）快路不成立。
  // cvProjectPoints2 忽略内参矩阵的斜切项，所以一并要求它为零，免得两条路在
  // 带斜切的标定上悄悄分叉——宁可慢，不可不一致。
  if (distortion.total() > 5 || matrix.at<double>(0, 1) != 0.0 ||
      object_points.size() != armor.points.size()) {
    return;
  }

  fx_ = matrix.at<double>(0, 0);
  fy_ = matrix.at<double>(1, 1);
  cx_ = matrix.at<double>(0, 2);
  cy_ = matrix.at<double>(1, 2);
  const auto coefficient = [&distortion](int index) {
    return index < static_cast<int>(distortion.total())
               ? distortion.at<double>(index)
               : 0.0;
  };
  k1_ = coefficient(0);
  k2_ = coefficient(1);
  p1_ = coefficient(2);
  p2_ = coefficient(3);
  k3_ = coefficient(4);

  // 半宽半高直接取自 object_points，连 armorPoints 收窄到 float 的那一步一起
  // 继承——慢路喂给 cv::projectPoints 的就是这几个数。
  half_width_ = object_points[0].y;
  half_height_ = object_points[0].z;

  const Eigen::Matrix3d R_camera_world =
      R_camera_barrel.transpose() * R_barrel_world.transpose();
  axis_x_ = R_camera_world.col(0);
  axis_y_ = R_camera_world.col(1);
  axis_z_ = R_camera_world.col(2);
  // 括号分组与 reproject_armor 保持一致，少一处无谓的浮点差异。
  center_ = R_camera_barrel.transpose() *
            (R_barrel_world.transpose() * armor.xyz_in_world - t_camera_barrel);

  const double pitch = armorPitchOf(armor.name);
  sin_pitch_ = std::sin(pitch);
  cos_pitch_ = std::cos(pitch);
  observed_ = armor.points;

  usable_ = R_camera_world.allFinite() && center_.allFinite() &&
            std::isfinite(fx_) && std::isfinite(fy_) && fx_ != 0.0 &&
            fy_ != 0.0;
}

double YawCostCache::cost(double yaw) const noexcept {
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  // armor -> world 的第二列是板面横向（与安装倾角无关），第三列是板面竖向。
  // 两者都已左乘过 world -> camera，所以直接就是相机系下的方向。
  const Eigen::Vector3d lateral = cos_yaw * axis_y_ - sin_yaw * axis_x_;
  const Eigen::Vector3d vertical =
      sin_pitch_ * (cos_yaw * axis_x_ + sin_yaw * axis_y_) +
      cos_pitch_ * axis_z_;
  const Eigen::Vector3d half_lateral = half_width_ * lateral;
  const Eigen::Vector3d half_vertical = half_height_ * vertical;

  // 顺序必须与 armorPoints 一致：左上、右上、右下、左下。
  const std::array<Eigen::Vector3d, 4> corners{
      center_ + half_lateral + half_vertical,
      center_ - half_lateral + half_vertical,
      center_ - half_lateral - half_vertical,
      center_ + half_lateral - half_vertical};

  double error = 0.0;
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const Eigen::Vector3d &point = corners[index];
    const double a = point.x() / point.z();
    const double b = point.y() / point.z();
    const double r2 = a * a + b * b;
    const double radial = 1.0 + r2 * (k1_ + r2 * (k2_ + r2 * k3_));
    const double xd = a * radial + 2.0 * p1_ * a * b + p2_ * (r2 + 2.0 * a * a);
    const double yd = b * radial + p1_ * (r2 + 2.0 * b * b) + 2.0 * p2_ * a * b;
    // 收窄成 float 再作差：慢路把投影点存进 std::vector<cv::Point2f>，那一步的
    // 舍入是两条路唯一的系统性差异来源，不复刻的话代价会差出 1e-4 px。
    const cv::Point2f projected{static_cast<float>(xd * fx_ + cx_),
                                static_cast<float>(yd * fy_ + cy_)};
    error += cv::norm(observed_[index] - projected);
  }
  return error;
}

bool
validCalibration(const L1Sensor::CameraCalibration &calibration) {
  // OpenCV 支持的常用畸变参数长度；拒绝形状虽合法但模型含义未知的数组。
  const std::size_t distortion_count =
      calibration.distortion_coefficients.total();
  const bool distortion_count_ok =
      distortion_count == 4 || distortion_count == 5 || distortion_count == 8 ||
      distortion_count == 12 || distortion_count == 14;

  if (calibration.image_size.width <= 0 || calibration.image_size.height <= 0 ||
      calibration.camera_matrix.type() != CV_64FC1 ||
      calibration.camera_matrix.rows != 3 ||
      calibration.camera_matrix.cols != 3 ||
      calibration.distortion_coefficients.type() != CV_64FC1 ||
      calibration.distortion_coefficients.empty() || !distortion_count_ok ||
      !cv::checkRange(calibration.camera_matrix) ||
      !cv::checkRange(calibration.distortion_coefficients) ||
      calibration.camera_matrix.at<double>(0, 0) <= 0.0 ||
      calibration.camera_matrix.at<double>(1, 1) <= 0.0 ||
      !calibration.T_barrel_camera) {
    return false;
  }

  const auto &transform = *calibration.T_barrel_camera;
  const Eigen::Matrix3d rotation = transform.linear();
  // 静态外参必须是有限的刚体变换，旋转块应接近正交且行列式为 1。
  constexpr double kRotationTolerance = 1e-3;
  return transform.matrix().allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                 .norm() <= kRotationTolerance &&
         std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

bool validConfig(const ArmorConfig &config) {
  // 几何尺寸必须为有限的正值。
  return std::isfinite(config.small_width) && config.small_width > 0.0 &&
         std::isfinite(config.big_width) && config.big_width > 0.0 &&
         std::isfinite(config.height) && config.height > 0.0;
}

void resetPnpOutput(Armor &armor) {
  // 保留 L2 输入字段，只清除由 PnP 生成的派生结果。
  armor.name = ArmorName::Unknown;
  armor.type = ArmorType::Small;
  armor.xyz_in_camera.setZero();
  armor.xyz_in_world.setZero();
  armor.ypr_in_camera.setZero();
  armor.ypr_in_world.setZero();
  armor.ypd_in_world.setZero();
  armor.reprojection_error = std::numeric_limits<double>::infinity();
}

// SP 对 3/4/5 号的大装甲（平衡步兵）跳过固定俯仰假设的 yaw 优化。当前板型映射
// 里平衡步兵已不存在，保留这条判据只为让单板与双板两条路径口径一致。
bool isBalanceInfantry(const Armor &armor) noexcept {
  return armor.type == ArmorType::Big &&
         (armor.name == ArmorName::Infantry3 ||
          armor.name == ArmorName::Infantry4 ||
          armor.name == ArmorName::Infantry5);
}

// 能否参与双板配对：single_pnp 必须已提交位姿（name 只在提交那一步被赋值），
// 类别必须能查到板数，且该类别的 yaw 优化没有被跳过。
bool pairableObservation(const Armor &armor) noexcept {
  return armor.name != ArmorName::Unknown && armor.xyz_in_world.allFinite() &&
         armorCountOf(armor.name).has_value() && !isBalanceInfantry(armor);
}

bool finiteImagePoints(const std::array<cv::Point2f, 4> &points) {
  // 在调用 OpenCV 前拦截 NaN 和无穷像素坐标。
  return std::all_of(points.begin(), points.end(),
                     [](const cv::Point2f &point) {
                       return std::isfinite(point.x) && std::isfinite(point.y);
                     });
}

} // namespace

PnpSolver::PnpSolver(const L1Sensor::CameraCalibration &calibration,
                     ArmorConfig config)
    : calibration_(calibration), config_(config) {
  small_armor_points_ = armorPoints(config_.small_width, config_.height);
  big_armor_points_ = armorPoints(config_.big_width, config_.height);
  ready_ = setCalibration(calibration);
}

bool PnpSolver::setCalibration(const L1Sensor::CameraCalibration &calibration) {
  ready_ = false;
  if (!validCalibration(calibration) || !validConfig(config_)) {
    return false;
  }

  // 缓存静态外参，避免逐帧从齐次变换中重复拆分。
  calibration_ = calibration;
  R_camera2barrel_ = calibration.T_barrel_camera->linear();
  t_camera2barrel_ = calibration.T_barrel_camera->translation();
  ready_ = R_camera2barrel_.allFinite() && t_camera2barrel_.allFinite();
  return ready_;
}

bool PnpSolver::ready() const noexcept { return ready_; }

void PnpSolver::set_R_world_barrel(
    const std::optional<Eigen::Quaterniond> &barrel_pose) {
  // 每帧先清除有效标志，缺失或退化四元数不能沿用上一帧姿态。
  world_barrel_ready_ = false;
  if (!barrel_pose || !barrel_pose->coeffs().allFinite() ||
      barrel_pose->squaredNorm() <= 1e-12) {
    return;
  }

  // SP 直接调用 q.toRotationMatrix()，不在 Solver 内再次归一化。
  R_barrel2world_ = barrel_pose->toRotationMatrix();
  world_barrel_ready_ = R_barrel2world_.allFinite();
}

void PnpSolver::single_pnp(Armor &armor) const {
  // Armor 可能被跨帧复用；任何提前返回都只能留下明确的无效输出。
  resetPnpOutput(armor);

  const auto armor_type = armorTypeFromClassId(armor.class_id);
  // PnP 同时依赖静态标定、曝光时刻姿态、有效类别和有限角点。
  if (!ready_ || !world_barrel_ready_ || !armor_type ||
      !finiteImagePoints(armor.points)) {
    return;
  }

  const auto &object_points = *armor_type == ArmorType::Big
    ? big_armor_points_
    : small_armor_points_;
  const std::vector<cv::Point2f> image_points(armor.points.begin(),
                                              armor.points.end());

  cv::Vec3d rvec;
  cv::Vec3d tvec;
  bool solved = false;
  // 平面 IPPE 使用四个有序角点恢复 armor -> camera 位姿。
  try {
    solved =
        cv::solvePnP(object_points, image_points, calibration_.camera_matrix,
                     calibration_.distortion_coefficients, rvec, tvec, false,
                     cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver solvePnP failed", error.what());
    return;
  }

  if (!solved) {
    return;
  }

  // 板中心必须在相机前方，旋转向量和平移向量均不得含非有限值。
  if (!std::isfinite(rvec[0]) || !std::isfinite(rvec[1]) ||
      !std::isfinite(rvec[2]) || !std::isfinite(tvec[0]) ||
      !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]) ||
      tvec[2] <= kMinimumCornerDepth) {
    return;
  }

  cv::Mat R_armor2camera_cv;
  std::vector<cv::Point2f> reprojected_points;
  // 同时生成旋转矩阵和重投影点，供坐标变换及像素误差计算复用。
  try {
    cv::Rodrigues(rvec, R_armor2camera_cv);
    cv::projectPoints(object_points, rvec, tvec, calibration_.camera_matrix,
                      calibration_.distortion_coefficients, reprojected_points);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver pose conversion failed", error.what());
    return;
  }

  if (!cv::checkRange(R_armor2camera_cv) ||
      reprojected_points.size() != image_points.size()) {
    return;
  }

  // 中心在相机前方并不足够，倾斜时四个物理角点也必须全部可见。
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(R_armor2camera_cv, R_armor2camera);
  for (const cv::Point3f &point : object_points) {
    const Eigen::Vector3d point_in_camera =
      R_armor2camera * Eigen::Vector3d{point.x, point.y, point.z} +
      Eigen::Vector3d{tvec[0], tvec[1], tvec[2]};
    if (!point_in_camera.allFinite() ||
        point_in_camera.z() <= kMinimumCornerDepth) {
      return;
    }
  }

  // 重投影 RMSE 只作为诊断量输出，不再作为观测门限——sp_vision 不计算它，
  // 也不据此拒绝任何观测。离线回放和遥测仍会读这个字段。
  // 取四角点二维欧氏误差的 RMSE，衡量的是 IPPE 这一步解得好不好，因此用
  // 相机系原始解算，和之后的 yaw 优化无关。
  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    const double dx = reprojected_points[index].x - image_points[index].x;
    const double dy = reprojected_points[index].y - image_points[index].y;
    squared_error_sum += dx * dx + dy * dy;
  }
  const double reprojection_error =
      std::sqrt(squared_error_sum / static_cast<double>(image_points.size()));

  // solvePnP 给出 armor -> camera；静态外参再转换到 barrel 和 world。
  const Eigen::Vector3d xyz_in_camera{tvec[0], tvec[1], tvec[2]};
  const Eigen::Vector3d xyz_in_barrel = R_camera2barrel_ * xyz_in_camera + t_camera2barrel_;
  const Eigen::Vector3d xyz_in_world = R_barrel2world_ * xyz_in_barrel;

  const Eigen::Matrix3d R_armor2barrel = R_camera2barrel_ * R_armor2camera;
  const Eigen::Matrix3d R_armor2world = R_barrel2world_ * R_armor2barrel;

  //注意这个是朝向角
  const Eigen::Vector3d ypr_in_camera = L6Telemetry::eulers(R_armor2camera, 2, 1, 0);
  const Eigen::Vector3d ypr_in_barrel = L6Telemetry::eulers(R_armor2barrel, 2, 1, 0);
  const Eigen::Vector3d ypr_in_world = L6Telemetry::eulers(R_armor2world, 2, 1, 0);
  //注意这个是方位角
  const Eigen::Vector3d ypd_in_world = L6Telemetry::xyz2ypd(xyz_in_world);

  // 上面 rvec/tvec 已验过有限，外参在 setCalibration / set_R_world_barrel 里也
  // 验过，其余量都是它们的乘积与 atan2，不会凭空变成非有限。这里只守住真正
  // 交给 EKF 的那两个——它们是本函数唯一的对外产物。
  if (!xyz_in_world.allFinite() || !ypr_in_world.allFinite()) {
    return;
  }

  // 所有派生量通过有限性检查后再一次性提交，避免暴露半成品观测。
  armor.name = L2Perception::armorClassFromId(armor.class_id);
  armor.type = *armor_type;
  //位置
  armor.xyz_in_camera = xyz_in_camera;
  armor.xyz_in_barrel = xyz_in_barrel;
  armor.xyz_in_world = xyz_in_world;
  //姿态
  armor.ypr_in_camera = ypr_in_camera;
  armor.ypr_in_barrel = ypr_in_barrel;
  armor.ypr_in_world = ypr_in_world;

  armor.ypd_in_world = ypd_in_world;
  armor.reprojection_error = reprojection_error;

  // SP 对 3/4/5 号的大装甲（平衡步兵）跳过固定俯仰假设的 yaw 优化。
  if (isBalanceInfantry(armor)) {
    return;
  }

  optimize_yaw(armor);
}

const std::vector<cv::Point3f> &
PnpSolver::objectPointsFor(ArmorType type) const noexcept {
  return type == ArmorType::Big ? big_armor_points_ : small_armor_points_;
}

double PnpSolver::yaw_cost(const Armor &armor, double yaw) const {
  const std::vector<cv::Point2f> projected =
    reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  if (projected.size() != armor.points.size()) {
    // 与 SP 的 continue 等价：无穷代价永远不会赢得比较。
    return std::numeric_limits<double>::infinity();
  }

  double error = 0.0;
  for (std::size_t point = 0; point < projected.size(); ++point) {
    error += cv::norm(armor.points[point] - projected[point]);
  }
  return error;
}

void PnpSolver::optimize_yaw(Armor &armor) const {
  // 以下搜索范围、步长、代价与平局规则逐项对应 SP Solver::optimize_yaw。
  const double barrel_yaw =
    L6Telemetry::eulers(R_barrel2world_, 2, 1, 0)[0];
  const double yaw0 = spLimitRad(
    barrel_yaw - kYawSearchRangeDegrees / 2.0 * CV_PI / 180.0);

  // 这一帧这块板的重投影常量只算一次。标定不受支持时 usable() 为 false，
  // 整个搜索原样退回 yaw_cost，行为与快路不存在时完全一致。
  const YawCostCache cache{calibration_,     R_camera2barrel_,
                           t_camera2barrel_, R_barrel2world_,
                           objectPointsFor(armor.type), armor};
  const bool fast = cache.usable();

  // 整步的代价全部留下，细化要用胜者左右两格——它们已经算过，不必重算。
  std::array<double, kYawSearchSteps> costs{};
  double min_error = 1e10;
  int best_index = -1;
  double best_yaw = armor.ypr_in_world[0];
  for (int index = 0; index < kYawSearchSteps; ++index) {
    const double yaw = spLimitRad(yaw0 + index * CV_PI / 180.0);
    costs[index] = fast ? cache.cost(yaw) : yaw_cost(armor, yaw);
    if (costs[index] < min_error) {
      min_error = costs[index];
      best_yaw = yaw;
      best_index = index;
    }
  }

  // 胜者落在窗口两端时缺一侧邻居，不细化。那本来就是真解在 140 度窗口之外、
  // 输出被截断在边界上的情形，细化没有意义。best_index 为 -1 表示整窗代价
  // 全部非有限，同样跳过。
  if (best_index > 0 && best_index + 1 < kYawSearchSteps) {
    const double offset = parabolicYawOffset(
      costs[best_index - 1], costs[best_index], costs[best_index + 1]);
    best_yaw = spLimitRad(best_yaw + offset * CV_PI / 180.0);
  }

  armor.yaw_raw = armor.ypr_in_world[0];
  armor.yaw_sigma = std::numeric_limits<double>::infinity();
  armor.ypr_in_world[0] = best_yaw;
}

bool PnpSolver::optimize_yaw_pair(Armor &left, Armor &right) const {
  const std::optional<int> armor_count = armorCountOf(left.name);
  if (!armor_count) {
    return false;
  }
  // 相邻两块板的朝向差。四板车 90 度、前哨与基地 120 度——rm.cv.fans 在这里
  // 写死了 π/2，泛化成 2π/n 才对得上三板车。
  const double offset =
    2.0 * std::numbers::pi / static_cast<double>(*armor_count);

  // 窗口、步长、代价与单板搜索完全一致，只把代价换成两块板之和：不额外引入
  // 单峰假设，代价曲线仍能和单板逐点对比（离线回放据此画两条曲线）。
  const double barrel_yaw = L6Telemetry::eulers(R_barrel2world_, 2, 1, 0)[0];
  const double yaw0 = spLimitRad(
    barrel_yaw - kYawSearchRangeDegrees / 2.0 * CV_PI / 180.0);

  const YawCostCache left_cache{calibration_,     R_camera2barrel_,
                                t_camera2barrel_, R_barrel2world_,
                                objectPointsFor(left.type), left};
  const YawCostCache right_cache{calibration_,     R_camera2barrel_,
                                 t_camera2barrel_, R_barrel2world_,
                                 objectPointsFor(right.type), right};
  const bool fast = left_cache.usable() && right_cache.usable();

  std::array<double, kYawSearchSteps> costs{};
  double min_error = std::numeric_limits<double>::infinity();
  int best_index = -1;
  double best_left_yaw = left.ypr_in_world[0];
  for (int index = 0; index < kYawSearchSteps; ++index) {
    const double yaw = spLimitRad(yaw0 + index * CV_PI / 180.0);
    const double right_yaw = spLimitRad(yaw + offset);
    costs[index] = fast
      ? left_cache.cost(yaw) + right_cache.cost(right_yaw)
      : yaw_cost(left, yaw) + yaw_cost(right, right_yaw);
    if (costs[index] < min_error) {
      min_error = costs[index];
      best_left_yaw = yaw;
      best_index = index;
    }
  }
  if (!std::isfinite(min_error)) {
    return false;
  }

  // 联合代价是两块板代价之和，两项在极小点附近都光滑，和仍然局部二次，
  // 细化的前提与单板一样成立。右板 yaw 由 2π/n 的约束跟着走。
  if (best_index > 0 && best_index + 1 < kYawSearchSteps) {
    const double refinement = parabolicYawOffset(
      costs[best_index - 1], costs[best_index], costs[best_index + 1]);
    best_left_yaw = spLimitRad(best_left_yaw + refinement * CV_PI / 180.0);
  }

  // rm.cv.fans 和 QD 在这里都不设代价门限：配对一旦成立就无条件采用联合解，
  // 可信度由配对条件本身保证（同类别、同板型、间距在相邻板的物理范围内）。
  left.ypr_in_world[0] = best_left_yaw;
  right.ypr_in_world[0] = spLimitRad(best_left_yaw + offset);
  return true;
}

void PnpSolver::refine_double_armor(std::vector<Armor> &armors) const {
  if (!ready_ || !world_barrel_ready_ || armors.size() < 2) {
    return;
  }

  // 同一车辆类别在场上只有一个实体（前哨、基地同理），因此同 name 同板型的
  // 两块板就是同一辆车的相邻两板。同类别出现三块以上时只取世界系最近的一对，
  // 多出来的那块必然是误检——rm.cv.fans 在这种情况下直接不修正 yaw。
  std::vector<bool> paired(armors.size(), false);
  for (std::size_t index = 0; index < armors.size(); ++index) {
    if (paired[index] || !pairableObservation(armors[index])) {
      continue;
    }

    std::size_t partner = armors.size();
    double partner_gap = kMaximumPairGap;
    for (std::size_t other = index + 1; other < armors.size(); ++other) {
      if (paired[other] || !pairableObservation(armors[other]) ||
          armors[other].name != armors[index].name ||
          armors[other].type != armors[index].type) {
        continue;
      }

      const double gap =
        (armors[index].xyz_in_world - armors[other].xyz_in_world).norm();
      // 下限拦重复检测：同一块板被检出两次时间距接近 0，按相邻板配对会凭空
      // 造出一个 2π/n 的约束。
      if (gap < kMinimumPairGap || gap >= partner_gap) {
        continue;
      }
      partner_gap = gap;
      partner = other;
    }

    if (partner >= armors.size()) {
      continue;
    }

    // 世界系 y 指左，方位角大的那块板在图像左侧。用方位角而不是像素横坐标
    // 排序，左右判定就不受相机 roll 影响。
    const bool index_is_left =
      armors[index].ypd_in_world[0] >= armors[partner].ypd_in_world[0];
    Armor &left = index_is_left ? armors[index] : armors[partner];
    Armor &right = index_is_left ? armors[partner] : armors[index];
    if (optimize_yaw_pair(left, right)) {
      paired[index] = true;
      paired[partner] = true;
    }
  }
}

std::vector<cv::Point2f>
PnpSolver::reproject_armor(const Eigen::Vector3d &xyz_in_world, double yaw,
                           ArmorType type, ArmorName name) const {
  if (!ready_ || !world_barrel_ready_ || !xyz_in_world.allFinite() ||
      !std::isfinite(yaw)) {
    return {};
  }

  const Eigen::Matrix3d R_armor2world = armorRotationInWorld(yaw, name);
  const Eigen::Matrix3d R_armor2camera =
    R_camera2barrel_.transpose() * R_barrel2world_.transpose() *
    R_armor2world;
  const Eigen::Vector3d t_armor2camera =
    R_camera2barrel_.transpose() *
    (R_barrel2world_.transpose() * xyz_in_world - t_camera2barrel_);
  if (!R_armor2camera.allFinite() || !t_armor2camera.allFinite()) {
    return {};
  }

  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Vec3d rvec;
  cv::Vec3d tvec{
    t_armor2camera.x(), t_armor2camera.y(), t_armor2camera.z()};
  std::vector<cv::Point2f> image_points;
  try {
    cv::Rodrigues(R_armor2camera_cv, rvec);
    cv::projectPoints(
      objectPointsFor(type), rvec, tvec, calibration_.camera_matrix,
      calibration_.distortion_coefficients, image_points);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver armor reprojection failed", error.what());
    return {};
  }
  return image_points;
}

} // namespace L3Estimation
