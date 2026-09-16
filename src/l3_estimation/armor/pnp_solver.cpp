#include "l3_estimation/armor/pnp_solver.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace L3Estimation {
namespace {

// 角点深度小于该值时视为落在相机平面或相机后方。
constexpr double kMinimumCornerDepth = 1e-6;

// optimize_yaw 的搜索窗口宽度，单位为度，步长 1 度。
constexpr double kYawSearchRangeDegrees = 140.0;

// 黄金分割搜一元函数的极小点，供前哨的固定俯仰 yaw 修正使用：每轮按收缩
// 系数缩短区间，直到长度小于终止精度。
template <typename Function>
double goldenSectionSearch(
  Function function, double left, double right, double epsilon = 1e-4)
{
  constexpr double kPhi = 0.6180339887498948482;
  double x1 = right - kPhi * (right - left);
  double x2 = left + kPhi * (right - left);
  double f1 = function(x1);
  double f2 = function(x2);

  while (right - left > epsilon) {
    if (f1 > f2) {
      left = x1;
      x1 = x2;
      f1 = f2;
      x2 = left + kPhi * (right - left);
      f2 = function(x2);
    } else {
      right = x2;
      x2 = x1;
      f2 = f1;
      x1 = right - kPhi * (right - left);
      f1 = function(x1);
    }
  }
  return f1 < f2 ? x1 : x2;
}

// class_id → 板型，用来选 PnP 物点尺寸；未知类别返回 nullopt，不参与求解。
// 映射本身在 types.hpp，与 L5 火控共用一份，免得两处各写一遍后悄悄分叉。
constexpr std::optional<ArmorType>
armorTypeFromClassId(int class_id) noexcept {
  return armorTypeOf(L2Perception::armorClassFromId(class_id));
}

// 生成一块板的四个物点，顺序与图像角点一致（左上、右上、右下、左下）。
// 构造时算好存起来，yaw 搜索那 140 次重投影就不用反复分配。
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

// 由世界系 yaw 拼出 armor -> world 旋转：俯仰取该车型固定的安装倾角，只有
// yaw 是自由量，这正是 yaw 搜索成立的前提。
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
  // 只认 OpenCV 常用的几种畸变参数长度，形状合法但模型含义不明的一律拒绝。
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
  // 静态外参必须是有限的刚体变换：旋转块接近正交，行列式为 +1。
  constexpr double kRotationTolerance = 1e-3;
  return transform.matrix().allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
                 .norm() <= kRotationTolerance &&
         std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
}

bool validConfig(const ArmorConfig &config) {
  // 三个几何尺寸都必须是有限正值。
  return std::isfinite(config.small_width) && config.small_width > 0.0 &&
         std::isfinite(config.big_width) && config.big_width > 0.0 &&
         std::isfinite(config.height) && config.height > 0.0;
}

void resetPnpOutput(Armor &armor) {
  // 只清 PnP 生成的派生结果，保留 L2 传进来的输入字段。
  armor.name = ArmorName::Unknown;
  armor.type = ArmorType::Small;
  armor.xyz_in_camera.setZero();
  armor.xyz_in_world.setZero();
  armor.ypr_in_camera.setZero();
  armor.ypr_in_world.setZero();
  armor.ypd_in_world.setZero();
  armor.reprojection_error = std::numeric_limits<double>::infinity();
}

// 3/4/5 号的大板（平衡步兵）不做固定俯仰假设的 yaw 优化。当前板型映射里平衡
// 步兵已不存在，这条判据恒为假，留着是为了换板型映射时不漏掉这一支。
bool isBalanceInfantry(const Armor &armor) noexcept {
  return armor.type == ArmorType::Big &&
         (armor.name == ArmorName::Infantry3 ||
          armor.name == ArmorName::Infantry4 ||
          armor.name == ArmorName::Infantry5);
}

bool finiteImagePoints(const std::array<cv::Point2f, 4> &points) {
  // 在交给 OpenCV 之前拦掉 NaN 和无穷像素坐标。
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

  // 把静态外参拆成 R 和 t 存下来，省得逐帧从齐次变换里再拆一遍。
  calibration_ = calibration;
  R_camera2barrel_ = calibration.T_barrel_camera->linear();
  t_camera2barrel_ = calibration.T_barrel_camera->translation();
  ready_ = R_camera2barrel_.allFinite() && t_camera2barrel_.allFinite();
  return ready_;
}

bool PnpSolver::ready() const noexcept { return ready_; }

void PnpSolver::set_R_world_barrel(
    const std::optional<Eigen::Quaterniond> &barrel_pose) {
  // 先清有效标志：传进来的四元数缺失或退化时，不能沿用上一帧的姿态。
  world_barrel_ready_ = false;
  if (!barrel_pose || !barrel_pose->coeffs().allFinite() ||
      barrel_pose->squaredNorm() <= 1e-12) {
    return;
  }

  // 直接取旋转矩阵，不在这里再归一化一次四元数。
  R_barrel2world_ = barrel_pose->toRotationMatrix();
  world_barrel_ready_ = R_barrel2world_.allFinite();
}

void PnpSolver::single_pnp(Armor &armor) const {
  // armor 可能被跨帧复用，先清空派生结果：后面任何提前返回留下的都是明确的
  // 无效输出。
  resetPnpOutput(armor);

  const auto armor_type = armorTypeFromClassId(armor.class_id);
  // 四个前提缺一不可：静态标定、曝光时刻的枪管姿态、有效类别、有限角点。
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
  // 平面 IPPE：用四个有序角点解出 armor -> camera 位姿。
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

  // 解出来的 rvec/tvec 要有限，板中心要在相机前方。
  if (!std::isfinite(rvec[0]) || !std::isfinite(rvec[1]) ||
      !std::isfinite(rvec[2]) || !std::isfinite(tvec[0]) ||
      !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]) ||
      tvec[2] <= kMinimumCornerDepth) {
    return;
  }

  cv::Mat R_armor2camera_cv;
  std::vector<cv::Point2f> reprojected_points;
  // 一次算出旋转矩阵和重投影点，后面的坐标变换与像素误差都用它们。
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

  // 中心在相机前方还不够：板倾斜时四个物点也必须都在相机前方。
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

  // 四个角点二维欧氏误差的 RMSE，用相机系的原始解算，衡量的是 IPPE 这一步
  // 解得好不好，与之后的 yaw 优化无关。它只作诊断输出，不作观测门限，离线
  // 回放和遥测会读这个字段。
  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    const double dx = reprojected_points[index].x - image_points[index].x;
    const double dy = reprojected_points[index].y - image_points[index].y;
    squared_error_sum += dx * dx + dy * dy;
  }
  const double reprojection_error =
      std::sqrt(squared_error_sum / static_cast<double>(image_points.size()));

  // solvePnP 给的是 armor -> camera，再用静态外参和枪管姿态转到 barrel、world。
  const Eigen::Vector3d xyz_in_camera{tvec[0], tvec[1], tvec[2]};
  const Eigen::Vector3d xyz_in_barrel = R_camera2barrel_ * xyz_in_camera + t_camera2barrel_;
  const Eigen::Vector3d xyz_in_world = R_barrel2world_ * xyz_in_barrel;

  const Eigen::Matrix3d R_armor2barrel = R_camera2barrel_ * R_armor2camera;
  const Eigen::Matrix3d R_armor2world = R_barrel2world_ * R_armor2barrel;

  // 注意这三个是朝向角 [yaw, pitch, roll]。
  const Eigen::Vector3d ypr_in_camera = L6Telemetry::eulers(R_armor2camera, 2, 1, 0);
  const Eigen::Vector3d ypr_in_barrel = L6Telemetry::eulers(R_armor2barrel, 2, 1, 0);
  const Eigen::Vector3d ypr_in_world = L6Telemetry::eulers(R_armor2world, 2, 1, 0);
  // 注意这个是方位角 [方位, 俯仰, 距离]。
  const Eigen::Vector3d ypd_in_world = L6Telemetry::xyz2ypd(xyz_in_world);

  // rvec/tvec 上面验过，外参在 setCalibration / set_R_world_barrel 里验过，
  // 其余量都是它们的乘积和 atan2，不会凭空变成非有限。所以这里只守真正交给
  // EKF 的那两个。
  if (!xyz_in_world.allFinite() || !ypr_in_world.allFinite()) {
    return;
  }

  // 全部通过检查后一次性写回 armor，不暴露半成品观测。
  armor.name = L2Perception::armorClassFromId(armor.class_id);
  armor.type = *armor_type;
  // 位置
  armor.xyz_in_camera = xyz_in_camera;
  armor.xyz_in_barrel = xyz_in_barrel;
  armor.xyz_in_world = xyz_in_world;
  // 姿态
  armor.ypr_in_camera = ypr_in_camera;
  armor.ypr_in_barrel = ypr_in_barrel;
  armor.ypr_in_world = ypr_in_world;

  armor.ypd_in_world = ypd_in_world;
  armor.reprojection_error = reprojection_error;

  // 平衡步兵跳过 yaw 优化，保留 IPPE 的原始姿态。
  if (isBalanceInfantry(armor)) {
    return;
  }

  optimize_yaw(armor);
}

std::optional<double> PnpSolver::lights_depth_diff(
  const Armor& armor) const
{
  const auto armor_type = armorTypeFromClassId(armor.class_id);
  const ArmorName name = L2Perception::armorClassFromId(armor.class_id);
  if (!ready_ || !armor_type || !finiteImagePoints(armor.points)) {
    return std::nullopt;
  }
  // 前哨的固定俯仰修正定义在世界系：没有曝光时刻的枪管姿态就直接返回空，
  // 不能悄悄退回另一种观测口径。
  if (name == ArmorName::Outpost && !world_barrel_ready_) {
    return std::nullopt;
  }

  const auto& object_points = *armor_type == ArmorType::Big
    ? big_armor_points_
    : small_armor_points_;
  const std::vector<cv::Point2f> image_points(
    armor.points.begin(), armor.points.end());

  std::vector<cv::Mat> rotation_vectors;
  std::vector<cv::Mat> translation_vectors;
  bool solved = false;
  try {
    solved = cv::solvePnPGeneric(
      object_points, image_points, calibration_.camera_matrix,
      calibration_.distortion_coefficients, rotation_vectors,
      translation_vectors, false, cv::SOLVEPNP_IPPE, cv::noArray(),
      cv::noArray());
  } catch (const cv::Exception& error) {
    L6Telemetry::logWarn(
      "PnpSolver depth-difference solvePnP failed", error.what());
    return std::nullopt;
  }
  if (!solved || rotation_vectors.size() != translation_vectors.size()) {
    return std::nullopt;
  }

  for (std::size_t index = 0; index < rotation_vectors.size(); ++index) {
    cv::Mat rotation_cv;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    try {
      cv::Rodrigues(rotation_vectors[index], rotation_cv);
      cv::cv2eigen(rotation_cv, rotation);
      cv::cv2eigen(translation_vectors[index], translation);
    } catch (const cv::Exception&) {
      continue;
    }
    if (!rotation.allFinite() || !translation.allFinite()) {
      continue;
    }

    // 板系 +x 指向车心，朝外的正面法向是 -x。IPPE 的候选已按重投影误差排序，
    // 取第一个正面朝向相机的即可。
    const Eigen::Vector3d front_normal = -rotation.col(0);
    if (front_normal.dot(-translation) <= 0.0) {
      continue;
    }

    // 前哨把姿态限制成固定 -15° 俯仰，并在 IPPE 给的世界系 yaw 左右各 70° 内
    // 用黄金分割重搜；普通车辆保留 IPPE 的原始姿态。
    Eigen::Matrix3d depth_rotation = rotation;
    if (name == ArmorName::Outpost) {
      const Eigen::Matrix3d rotation_in_world =
        R_barrel2world_ * R_camera2barrel_ * rotation;
      const Eigen::Vector3d translation_in_world =
        R_barrel2world_ *
        (R_camera2barrel_ * translation + t_camera2barrel_);
      const double raw_yaw =
        L6Telemetry::eulers(rotation_in_world, 2, 1, 0)[0];
      constexpr double kHalfRange =
        kYawSearchRangeDegrees * 0.5 * CV_PI / 180.0;
      const cv::Point2f measured_top =
        (armor.points[0] + armor.points[1]) * 0.5F;
      const cv::Point2f measured_bottom =
        (armor.points[3] + armor.points[2]) * 0.5F;
      const auto yawCost = [&](double yaw) {
        const auto projected = reproject_armor(
          translation_in_world, yaw, *armor_type, name);
        if (projected.size() != armor.points.size()) {
          return std::numeric_limits<double>::infinity();
        }
        const cv::Point2f projected_top =
          (projected[0] + projected[1]) * 0.5F;
        const cv::Point2f projected_bottom =
          (projected[3] + projected[2]) * 0.5F;
        return cv::norm(projected_top - measured_top) +
               cv::norm(projected_bottom - measured_bottom);
      };
      const double best_yaw = goldenSectionSearch(
        yawCost, raw_yaw - kHalfRange, raw_yaw + kHalfRange);
      depth_rotation =
        R_camera2barrel_.transpose() * R_barrel2world_.transpose() *
        armorRotationInWorld(best_yaw, name);
    }

    const auto pointInCamera = [&](std::size_t point_index) {
      const cv::Point3f& point = object_points[point_index];
      return (depth_rotation * Eigen::Vector3d(point.x, point.y, point.z) +
              translation).eval();
    };
    // 角点顺序为 TL、TR、BR、BL。
    const Eigen::Vector3d left_center =
      (pointInCamera(0) + pointInCamera(3)) * 0.5;
    const Eigen::Vector3d right_center =
      (pointInCamera(1) + pointInCamera(2)) * 0.5;
    const double difference = left_center.z() - right_center.z();
    if (std::isfinite(difference)) {
      return difference;
    }
  }
  return std::nullopt;
}

double PnpSolver::yaw_cost(const Armor &armor, double yaw) const {
  const std::vector<cv::Point2f> projected =
    reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  if (projected.size() != armor.points.size()) {
    // 返回无穷大：这个采样点永远赢不了比较，等价于直接跳过。
    return std::numeric_limits<double>::infinity();
  }

  double error = 0.0;
  for (std::size_t point = 0; point < projected.size(); ++point) {
    error += cv::norm(armor.points[point] - projected[point]);
  }
  return error;
}

void PnpSolver::optimize_yaw(Armor &armor) const {
  // 以枪管 yaw 为中心、左右各 70° 按 1° 枚举，取代价最小的一个；代价相同时
  // 保留先遇到的。
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
  armor.ypr_in_world[0] = best_yaw;
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
