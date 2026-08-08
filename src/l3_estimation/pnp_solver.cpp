#include "l3_estimation/pnp_solver.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace L3Estimation {
namespace {

// 角点深度小于该值时视为落在相机平面或相机后方。
constexpr double kMinimumCornerDepth = 1e-6;

// 将识别类别映射为实际 PnP 几何尺寸；未知类别不参与求解。映射本身放在
// types.hpp，与 L5 火控共用同一份，避免两处各写一遍后悄悄分叉。
[[nodiscard]] constexpr std::optional<ArmorType>
armorTypeFromClassId(int class_id) noexcept {
  return armorTypeOf(L2Perception::armorClassFromId(class_id));
}

[[nodiscard]] std::vector<cv::Point3d> armorPoints(ArmorType type,
                                                   const ArmorConfig &config) {
  const double half_width =
      (type == ArmorType::Big ? config.big_width : config.small_width) / 2.0;
  const double half_height = config.height / 2.0;

  // 装甲板局部坐标系中 x=0，四点顺序对应图像中的左上、右上、右下、左下。
  return {{0.0, half_width, half_height},
          {0.0, -half_width, half_height},
          {0.0, -half_width, -half_height},
          {0.0, half_width, -half_height}};
}

[[nodiscard]] bool
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

[[nodiscard]] bool validConfig(const ArmorConfig &config) {
  // 几何尺寸和质量门限必须为有限的正值；面积门限允许显式关闭。
  return std::isfinite(config.small_width) && config.small_width > 0.0 &&
         std::isfinite(config.big_width) && config.big_width > 0.0 &&
         std::isfinite(config.height) && config.height > 0.0 &&
         std::isfinite(config.max_reprojection_error) &&
         config.max_reprojection_error > 0.0 &&
         std::isfinite(config.min_area) && config.min_area >= 0.0;
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
  armor.quality = {};
}

[[nodiscard]] bool finiteImagePoints(const std::array<cv::Point2f, 4> &points) {
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

  // 输入四元数可能存在小幅数值漂移，转换前统一归一化。
  R_barrel2world_ = barrel_pose->normalized().toRotationMatrix();
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

  const auto object_points = armorPoints(*armor_type, config_);
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

  cv::Matx33d R_armor2camera_cv; 
  cv::Rodrigues(rvec, R_armor2camera_cv);

  // solvePnP 给出 armor -> camera；静态外参再转换到 barrel 和 world。
  const Eigen::Vector3d xyz_in_camera{tvec[0], tvec[1], tvec[2]};
  const Eigen::Vector3d xyz_in_barrel = R_camera2barrel_ * xyz_in_camera + t_camera2barrel_;
  const Eigen::Vector3d xyz_in_world = R_barrel2world_ * xyz_in_barrel;

  const Eigen::Matrix3d R_armor2camera = L6Telemetry::toEigen(R_armor2camera_cv);
  const Eigen::Matrix3d R_armor2barrel = R_camera2barrel_ * R_armor2camera;
  const Eigen::Matrix3d R_armor2world = R_barrel2world_ * R_armor2barrel;

  //注意这个是朝向角
  const Eigen::Vector3d ypr_in_camera = L6Telemetry::eulers(R_armor2camera, 2, 1, 0);
  const Eigen::Vector3d ypr_in_barrel = L6Telemetry::eulers(R_armor2barrel, 2, 1, 0);
  const Eigen::Vector3d ypr_in_world = L6Telemetry::eulers(R_armor2world, 2, 1, 0);
  //注意这个是方位角
  const Eigen::Vector3d ypd_in_world = L6Telemetry::xyz2ypd(xyz_in_world);

  const bool finite = xyz_in_camera.allFinite() && xyz_in_barrel.allFinite() &&
                      xyz_in_world.allFinite() && R_armor2camera.allFinite() &&
                      R_armor2world.allFinite() && ypr_in_camera.allFinite() &&
                      ypr_in_world.allFinite() && ypd_in_world.allFinite();
  if (!finite) {
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

  // yaw 优化对每块装甲板都执行。
  optimize_yaw(armor);
}

void PnpSolver::optimize_yaw(Armor &armor) const {

  constexpr double SEARCH_RANGE = 140;  // degree

  auto yaw0 = L6Telemetry::eulers(R_barrel2world_, 2, 1, 0);

  auto best_yaw = armor.ypr_in_world[0];

  double epsilon = 1e-3;
  double right = CV_PI/2;
  double left = -CV_PI/2;

  for (int iter = 0; right - left > epsilon; ++iter) {
    double mid1 = left + (right - left) / 3;
    double mid2 = right - (right - left) / 3;
    
    double f1 = armor_reprojection_error(armor, mid1);
    double f2 = armor_reprojection_error(armor, mid2);
    
    if (f1 < f2) {
        right = mid2;
    } else {
        left = mid1;
    }
  }
    
  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = (left + right) / 2;
}

double PnpSolver::armor_reprojection_error(const Armor &armor,
                                           double yaw) const {
  const std::vector<cv::Point2f> image_points =
      reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  if (image_points.size() != armor.points.size()) {
    return std::numeric_limits<double>::infinity();
  }

  // 搜索代价使用四个对应角点的像素距离之和。
  double error = 0.0;
  for (std::size_t index = 0; index < armor.points.size(); ++index) {
    error += cv::norm(armor.points[index] - image_points[index]);
  }
  return error;
}

std::vector<cv::Point2f>
PnpSolver::reproject_armor(const Eigen::Vector3d &xyz_in_world, double yaw,
                           ArmorType type, ArmorName name) const {
  if (!ready_ || !world_barrel_ready_ || !xyz_in_world.allFinite() ||
      !std::isfinite(yaw)) {
    return {};
  }
  // 根据车辆类别采用固定安装俯仰角，yaw 由调用方给出。
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  const double pitch = name == ArmorName::Outpost
                           ? -15.0 * std::numbers::pi / 180.0
                           : 15.0 * std::numbers::pi / 180.0;
  const double sin_pitch = std::sin(pitch);
  const double cos_pitch = std::cos(pitch);

  const Eigen::Matrix3d R_armor2world{
      {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
      {sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch},
      {-sin_pitch,              0,            cos_pitch}};

  // 依次应用 world -> barrel 和 barrel -> camera 逆变换。
  const Eigen::Matrix3d R_armor2camera = R_camera2barrel_.transpose() *
                                         R_barrel2world_.transpose() *
                                         R_armor2world;
  const Eigen::Vector3d t_armor2camera =
      R_camera2barrel_.transpose() *
      (R_barrel2world_.transpose() * xyz_in_world - t_camera2barrel_);
  if (!R_armor2camera.allFinite() || !t_armor2camera.allFinite()) {
    return {};
  }

  // 重投影前再次检查四个角点深度，避免生成无意义像素坐标。
  const auto object_points = armorPoints(type, config_);
  for (const cv::Point3d &point : object_points) {
    const Eigen::Vector3d point_in_camera =
        R_armor2camera * Eigen::Vector3d{point.x, point.y, point.z} +
        t_armor2camera;
    if (!point_in_camera.allFinite() ||
        point_in_camera.z() <= kMinimumCornerDepth) {
      return {};
    }
  }

  const cv::Vec3d tvec{t_armor2camera.x(), t_armor2camera.y(), t_armor2camera.z()};
  std::vector<cv::Point2d> projected_points;
  // OpenCV projectPoints 接收 Rodrigues 旋转向量，因此先转换矩阵表示。
  try {
    cv::Vec3d rvec;
    const cv::Matx33d R_armor2camera_cv = L6Telemetry::toCv(R_armor2camera);
    cv::Rodrigues(R_armor2camera_cv, rvec);
    cv::projectPoints(object_points, rvec, tvec, calibration_.camera_matrix,
                      calibration_.distortion_coefficients, projected_points);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver armor reprojection failed", error.what());
    return {};
  }

  std::vector<cv::Point2f> image_points;
  image_points.reserve(projected_points.size());
  for (const cv::Point2d &point : projected_points) {
    image_points.emplace_back(point);
  }
  return image_points;
}

} // namespace L3Estimation
