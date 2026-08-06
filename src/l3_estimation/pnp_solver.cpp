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

// 将识别类别映射为实际 PnP 几何尺寸；未知类别不参与求解。
// 场上只有四板车，大装甲板仅英雄使用：平衡步兵已不存在，基地虽然有
// Bs/Bb 两个类别但装甲板实物都是小板，所以只有 Hero 走 Big 分支。
[[nodiscard]] constexpr std::optional<ArmorType>
armorTypeFromClassId(int class_id) noexcept {
  using L2Perception::ArmorClass;

  switch (L2Perception::armorClassFromId(class_id)) {
  case ArmorClass::Hero:
    return ArmorType::Big;

  case ArmorClass::Guard:
  case ArmorClass::Engineer:
  case ArmorClass::Infantry3:
  case ArmorClass::Infantry4:
  case ArmorClass::Infantry5:
  case ArmorClass::Outpost:
  case ArmorClass::BaseSmall:
  case ArmorClass::BaseLarge:
    return ArmorType::Small;

  case ArmorClass::Unknown:
    break;
  }
  return std::nullopt;
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
  // pnp_ok 只表示求解器成功返回，后续仍需数值和几何检查。
  armor.quality.pnp_ok = true;

  // 板中心必须在相机前方，旋转向量和平移向量均不得含非有限值。
  if (!std::isfinite(rvec[0]) || !std::isfinite(rvec[1]) ||
      !std::isfinite(rvec[2]) || !std::isfinite(tvec[0]) ||
      !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]) ||
      tvec[2] <= kMinimumCornerDepth) {
    return;
  }

  cv::Matx33d R_armor2camera_cv;
  std::vector<cv::Point2d> reprojected_points;
  // 同时生成旋转矩阵和重投影点，供坐标变换及像素误差计算复用。
  try {
    cv::Rodrigues(rvec, R_armor2camera_cv);
    cv::projectPoints(object_points, rvec, tvec, calibration_.camera_matrix,
                      calibration_.distortion_coefficients, reprojected_points);
  } catch (const cv::Exception &error) {
    L6Telemetry::logWarn("PnpSolver pose conversion failed", error.what());
    return;
  }

  if (!cv::checkRange(cv::Mat(R_armor2camera_cv)) ||
      reprojected_points.size() != image_points.size()) {
    return;
  }

  // 中心在相机前方并不足够，倾斜时四个物理角点也必须全部可见。
  for (const cv::Point3d &point : object_points) {
    const cv::Vec3d point_in_camera =
        R_armor2camera_cv * cv::Vec3d{point.x, point.y, point.z} + tvec;
    if (!std::isfinite(point_in_camera[0]) ||
        !std::isfinite(point_in_camera[1]) ||
        !std::isfinite(point_in_camera[2]) ||
        point_in_camera[2] <= kMinimumCornerDepth) {
      return;
    }
  }

  armor.quality.geometry_ok = true;

  // 使用四角点二维欧氏误差的 RMSE 作为观测质量指标。
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
  const Eigen::Vector3d xyz_in_barrel =
      R_camera2barrel_ * xyz_in_camera + t_camera2barrel_;
  const Eigen::Vector3d xyz_in_world = R_barrel2world_ * xyz_in_barrel;

  const Eigen::Matrix3d R_armor2camera =
      L6Telemetry::toEigen(R_armor2camera_cv);
  const Eigen::Matrix3d R_armor2barrel = R_camera2barrel_ * R_armor2camera;
  const Eigen::Matrix3d R_armor2world = R_barrel2world_ * R_armor2barrel;
  const Eigen::Vector3d ypr_in_camera =
      L6Telemetry::eulers(R_armor2camera, 2, 1, 0);
  const Eigen::Vector3d ypr_in_world =
      L6Telemetry::eulers(R_armor2world, 2, 1, 0);
  const Eigen::Vector3d ypd_in_world = L6Telemetry::xyz2ypd(xyz_in_world);

  const bool finite = xyz_in_camera.allFinite() && xyz_in_barrel.allFinite() &&
                      xyz_in_world.allFinite() && R_armor2camera.allFinite() &&
                      R_armor2world.allFinite() && ypr_in_camera.allFinite() &&
                      ypr_in_world.allFinite() && ypd_in_world.allFinite() &&
                      std::isfinite(reprojection_error);
  if (!finite) {
    return;
  }

  // 所有派生量通过有限性检查后再一次性提交，避免暴露半成品观测。
  armor.name = L2Perception::armorClassFromId(armor.class_id);
  armor.type = *armor_type;
  armor.xyz_in_camera = xyz_in_camera;
  armor.xyz_in_world = xyz_in_world;
  armor.ypr_in_camera = ypr_in_camera;
  armor.ypr_in_world = ypr_in_world;
  armor.ypd_in_world = ypd_in_world;
  armor.reprojection_error = reprojection_error;
  armor.quality.finite = true;
  armor.quality.reprojection_ok =
      reprojection_error <= config_.max_reprojection_error;

  // 场上所有车辆都满足 reproject_armor 的固定安装倾角假设，
  // 因此 yaw 优化对每块装甲板都执行。
  optimize_yaw(armor);
}

void PnpSolver::optimize_yaw(Armor &armor) const {
  // 代价曲线在整个搜索范围内并非单峰：实测约四分之一的帧存在两个局部极小，
  // 典型间隔 70 度，来源是平面四点 PnP 的二义性。直接对全区间做黄金分割会
  // 收敛到其中任意一个，因此先用粗网格锁定全局极小所在的谷，再在相邻两格
  // 构成的区间内用黄金分割细化。
  //
  // 搜索在以枪管 yaw 为中心的展开区间内进行；代价只依赖 sin/cos，对 2*pi
  // 平移不变，所以中途无需归一化，只在写回时归一化一次。
  // 粗步长按回放实测选定：5 度时有约 0.4% 的帧会和逐度暴力搜索选到不同的
  // 谷（都是两个极小代价相差不到 4% 的近简并情形），2.5 度则完全一致。
  constexpr double kSearchRangeDegrees = 140.0;
  constexpr double kCoarseStepDegrees = 2.5;
  constexpr int kCoarseIntervals =
      static_cast<int>(kSearchRangeDegrees / kCoarseStepDegrees);
  // 细化到 0.03 度即可，远小于观测噪声，再细没有意义只会多算几次。
  constexpr double kRefineToleranceDegrees = 0.03;
  constexpr double kDegreeToRadian = std::numbers::pi / 180.0;

  const Eigen::Vector3d barrel_ypr =
      L6Telemetry::eulers(R_barrel2world_, 2, 1, 0);
  const double barrel_yaw = barrel_ypr[0];
  const double coarse_step = kCoarseStepDegrees * kDegreeToRadian;
  const double yaw0 = barrel_yaw - kSearchRangeDegrees * 0.5 * kDegreeToRadian;

  // 第一阶段：粗网格。只找全局最小所在的格点，不追求精度。
  int best_index = -1;
  double minimum_error = std::numeric_limits<double>::infinity();
  for (int index = 0; index <= kCoarseIntervals; ++index) {
    const double yaw = yaw0 + static_cast<double>(index) * coarse_step;
    const double error = armor_reprojection_error(armor, yaw);
    if (error < minimum_error) {
      minimum_error = error;
      best_index = index;
    }
  }

  // 全部格点都无法重投影时保持原 yaw，与逐度搜索的退化行为一致。
  if (best_index < 0 || !std::isfinite(minimum_error)) {
    return;
  }

  // 第二阶段：在最优格点两侧各扩一格作为搜索区间。谷宽至少两个粗步长时，
  // 该区间必定包住真正的极小，且区间内单峰，黄金分割才成立。
  double lower =
      yaw0 + static_cast<double>(std::max(best_index - 1, 0)) * coarse_step;
  double upper =
      yaw0 +
      static_cast<double>(std::min(best_index + 1, kCoarseIntervals)) *
          coarse_step;

  // 0.618...，每次迭代只需一次新的代价评估。
  constexpr double kInverseGoldenRatio = 0.6180339887498949;
  const double tolerance = kRefineToleranceDegrees * kDegreeToRadian;

  double probe_low = upper - kInverseGoldenRatio * (upper - lower);
  double probe_high = lower + kInverseGoldenRatio * (upper - lower);
  double error_low = armor_reprojection_error(armor, probe_low);
  double error_high = armor_reprojection_error(armor, probe_high);

  while (upper - lower > tolerance) {
    if (error_low < error_high) {
      upper = probe_high;
      probe_high = probe_low;
      error_high = error_low;
      probe_low = upper - kInverseGoldenRatio * (upper - lower);
      error_low = armor_reprojection_error(armor, probe_low);
    } else {
      lower = probe_low;
      probe_low = probe_high;
      error_low = error_high;
      probe_high = lower + kInverseGoldenRatio * (upper - lower);
      error_high = armor_reprojection_error(armor, probe_high);
    }
  }

  // 细化结果可能落在重投影失败的区域，此时退回粗网格上的最优格点。
  const double refined_yaw = 0.5 * (lower + upper);
  const double refined_error = armor_reprojection_error(armor, refined_yaw);
  const double coarse_best_yaw =
      yaw0 + static_cast<double>(best_index) * coarse_step;
  armor.ypr_in_world[0] = L6Telemetry::limit_rad(
      refined_error <= minimum_error ? refined_yaw : coarse_best_yaw);
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

  const cv::Vec3d tvec{t_armor2camera.x(), t_armor2camera.y(),
                       t_armor2camera.z()};
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
