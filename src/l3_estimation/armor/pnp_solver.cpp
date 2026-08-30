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

  double min_error = 1e10;
  double best_yaw = armor.ypr_in_world[0];
  for (int index = 0; index < kYawSearchRangeDegrees; ++index) {
    const double yaw = spLimitRad(yaw0 + index * CV_PI / 180.0);
    const double error = yaw_cost(armor, yaw);
    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
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

  double min_error = std::numeric_limits<double>::infinity();
  double best_left_yaw = left.ypr_in_world[0];
  for (int index = 0; index < kYawSearchRangeDegrees; ++index) {
    const double yaw = spLimitRad(yaw0 + index * CV_PI / 180.0);
    const double error =
      yaw_cost(left, yaw) + yaw_cost(right, spLimitRad(yaw + offset));
    if (error < min_error) {
      min_error = error;
      best_left_yaw = yaw;
    }
  }
  if (!std::isfinite(min_error)) {
    return false;
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
    const auto &object_points = type == ArmorType::Big
      ? big_armor_points_
      : small_armor_points_;
    cv::projectPoints(
      object_points, rvec, tvec, calibration_.camera_matrix,
      calibration_.distortion_coefficients, image_points);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver armor reprojection failed", error.what());
    return {};
  }
  return image_points;
}

} // namespace L3Estimation
