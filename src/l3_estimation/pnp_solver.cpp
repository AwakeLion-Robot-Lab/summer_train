#include "l3_estimation/pnp_solver.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace L3Estimation {
namespace {

// 装甲板四角点在板自身坐标系下的位置，顺序与 L2 的角点顺序一致：
// 左上、右上、右下、左下。板面法线为 x，宽度沿 y，高度沿 z。
[[nodiscard]] std::vector<cv::Point3f> armorPoints(double width, double height)
{
  const float half_width = static_cast<float>(width / 2.0);
  const float half_height = static_cast<float>(height / 2.0);
  return {
    {0.0F, half_width, half_height},
    {0.0F, -half_width, half_height},
    {0.0F, -half_width, -half_height},
    {0.0F, half_width, -half_height}};
}

// Huber 核：|e| <= delta 时是平方项，超出后转成线性，单个离群角点的影响被截断
// 在 delta 上而不是继续放大。
[[nodiscard]] double huberLoss(double error, double delta)
{
  const double absolute = std::abs(error);
  if (absolute <= delta) {
    return 0.5 * error * error;
  }
  return delta * (absolute - 0.5 * delta);
}

// 两个二维向量的夹角，单位 radian。任一向量退化成零长度时返回 0，表示这条边
// 提供不了方向信息；acos 的输入必须夹紧，浮点误差越界会直接变成 NaN 并污染
// 整条代价曲线。
[[nodiscard]] double vectorAngle(const cv::Point2d & lhs, const cv::Point2d & rhs)
{
  const double lhs_norm = std::hypot(lhs.x, lhs.y);
  const double rhs_norm = std::hypot(rhs.x, rhs.y);
  if (lhs_norm < 1e-9 || rhs_norm < 1e-9) {
    return 0.0;
  }

  const double cosine = (lhs.x * rhs.x + lhs.y * rhs.y) / (lhs_norm * rhs_norm);
  return std::acos(std::clamp(cosine, -1.0, 1.0));
}

}  // namespace

PnpSolver::PnpSolver(const L1Sensor::CameraCalibration & calibration, ArmorConfig config)
: config_(config),
  small_armor_points_(armorPoints(config.small_width, config.height)),
  big_armor_points_(armorPoints(config.big_width, config.height))
{
  ready_ = setCalibration(calibration);
}

bool PnpSolver::setCalibration(const L1Sensor::CameraCalibration & calibration)
{
  // 标定或装甲板尺寸不可用时保持 ready_ = false：绝不用单位阵或猜的尺寸顶替，
  // 那会让位姿静默地全错，比整条链路停下来难查得多。参数来自 YAML，所以这里
  // 必须挡住 NaN 和 0 这类填错的值。
  ready_ = false;
  if (config_.small_width <= 0.0 || config_.big_width <= 0.0 || config_.height <= 0.0) {
    return false;
  }
  if (!calibration.T_barrel_camera || calibration.camera_matrix.rows != 3 ||
      calibration.camera_matrix.cols != 3 || calibration.distortion_coefficients.empty() ||
      !cv::checkRange(calibration.camera_matrix) ||
      !cv::checkRange(calibration.distortion_coefficients) ||
      calibration.camera_matrix.at<double>(0, 0) <= 0.0 ||
      calibration.camera_matrix.at<double>(1, 1) <= 0.0 ||
      !calibration.T_barrel_camera->matrix().allFinite()) {
    return false;
  }

  calibration_ = calibration;
  R_camera2barrel_ = calibration.T_barrel_camera->linear();
  t_camera2barrel_ = calibration.T_barrel_camera->translation();
  ready_ = true;
  return true;
}

bool PnpSolver::ready() const noexcept { return ready_; }

void PnpSolver::set_R_world_barrel(const std::optional<Eigen::Quaterniond> & barrel_pose)
{
  // 每帧先清除有效标志：缺失或退化的四元数不能沿用上一帧姿态，那会把整条
  // 时间对齐链路悄悄错开一帧。
  world_barrel_ready_ = false;
  if (!barrel_pose || !barrel_pose->coeffs().allFinite() ||
      barrel_pose->squaredNorm() <= 1e-12) {
    return;
  }

  R_barrel2world_ = barrel_pose->toRotationMatrix();
  world_barrel_ready_ = true;
}

Eigen::Matrix3d PnpSolver::armorRotationInWorld(double yaw, ArmorName name) const
{
  // 装甲板按车型使用固定安装倾角，只有 yaw 是自由量——这正是 yaw 搜索成立的前提。
  const double pitch =
    name == ArmorName::Outpost ? config_.outpost_mount_pitch : config_.mount_pitch;
  const double sin_yaw = std::sin(yaw);
  const double cos_yaw = std::cos(yaw);
  const double sin_pitch = std::sin(pitch);
  const double cos_pitch = std::cos(pitch);

  return Eigen::Matrix3d{
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch},
    {-sin_pitch, 0, cos_pitch}};
}

void PnpSolver::single_pnp(Armor & armor) const
{
  // Armor 可能跨帧复用，先把 PnP 派生结果清空：任何提前返回都只会留下明确的
  // 无效输出，name 保持 Unknown 就是"本帧没有解出位姿"，Tracker 据此筛掉观测。
  armor.name = ArmorName::Unknown;
  armor.type = ArmorType::Small;
  armor.xyz_in_camera.setZero();
  armor.xyz_in_barrel.setZero();
  armor.xyz_in_world.setZero();
  armor.ypr_in_camera.setZero();
  armor.ypr_in_barrel.setZero();
  armor.ypr_in_world.setZero();
  armor.ypr_raw_in_world.setZero();
  armor.ypd_in_world.setZero();
  armor.reprojection_error = std::numeric_limits<double>::infinity();

  const auto armor_type = armorTypeOf(L2Perception::armorClassFromId(armor.class_id));
  // PnP 同时依赖静态标定、曝光时刻的枪管姿态和有效类别。
  if (!ready_ || !world_barrel_ready_ || !armor_type) {
    return;
  }

  const auto & object_points =
    *armor_type == ArmorType::Big ? big_armor_points_ : small_armor_points_;
  const std::vector<cv::Point2f> image_points(armor.points.begin(), armor.points.end());

  cv::Vec3d rvec;
  cv::Vec3d tvec;
  cv::Mat R_armor2camera_cv;
  std::vector<cv::Point2f> reprojected_points;
  try {
    // 平面 IPPE 由四个有序角点恢复 armor -> camera 位姿。
    if (!cv::solvePnP(
          object_points, image_points, calibration_.camera_matrix,
          calibration_.distortion_coefficients, rvec, tvec, false, cv::SOLVEPNP_IPPE)) {
      return;
    }
    cv::Rodrigues(rvec, R_armor2camera_cv);
    cv::projectPoints(
      object_points, rvec, tvec, calibration_.camera_matrix,
      calibration_.distortion_coefficients, reprojected_points);
  } catch (const cv::Exception & error) {
    // 单帧异常不允许打断主循环。
    L6Telemetry::logWarn("PnpSolver solvePnP failed", error.what());
    return;
  }

  // 解在相机后方说明角点顺序或标定有问题，这种位姿不能进 EKF。
  if (tvec[2] <= 0.0) {
    return;
  }

  // 四角点重投影 RMSE，取自相机系的原始解，与后面的 yaw 搜索无关，只作诊断。
  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    const double dx = reprojected_points[index].x - image_points[index].x;
    const double dy = reprojected_points[index].y - image_points[index].y;
    squared_error_sum += dx * dx + dy * dy;
  }

  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(R_armor2camera_cv, R_armor2camera);

  // camera -> barrel 是静态外参，barrel -> world 是曝光时刻的云台姿态。
  const Eigen::Vector3d xyz_in_camera{tvec[0], tvec[1], tvec[2]};
  const Eigen::Vector3d xyz_in_barrel = R_camera2barrel_ * xyz_in_camera + t_camera2barrel_;
  const Eigen::Vector3d xyz_in_world = R_barrel2world_ * xyz_in_barrel;
  const Eigen::Matrix3d R_armor2barrel = R_camera2barrel_ * R_armor2camera;
  // ypr 是装甲板的朝向角，ypd 是它在世界系下的方位角和距离。
  const Eigen::Vector3d ypr_in_world =
    L6Telemetry::eulers(R_barrel2world_ * R_armor2barrel, 2, 1, 0);

  if (!xyz_in_world.allFinite() || !ypr_in_world.allFinite()) {
    return;
  }

  // 全部算完再一次性提交，避免半成品位姿配上一个已提交的 name。
  armor.xyz_in_camera = xyz_in_camera;
  armor.xyz_in_barrel = xyz_in_barrel;
  armor.xyz_in_world = xyz_in_world;
  armor.ypr_in_camera = L6Telemetry::eulers(R_armor2camera, 2, 1, 0);
  armor.ypr_in_barrel = L6Telemetry::eulers(R_armor2barrel, 2, 1, 0);
  armor.ypr_in_world = ypr_in_world;
  armor.ypr_raw_in_world = ypr_in_world;
  armor.ypd_in_world = L6Telemetry::xyz2ypd(xyz_in_world);
  armor.reprojection_error =
    std::sqrt(squared_error_sum / static_cast<double>(image_points.size()));
  armor.name = L2Perception::armorClassFromId(armor.class_id);
  armor.type = *armor_type;

  optimize_yaw(armor);
}

double PnpSolver::armor_reprojection_error(const Armor & armor, double yaw) const
{
  const std::vector<cv::Point2f> projected =
    reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  if (projected.size() != armor.points.size() || projected.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const std::size_t count = projected.size();
  double cost = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    // 位置项：该角点的重投影像素距离。
    const cv::Point2f offset = armor.points[index] - projected[index];
    const double distance =
      std::hypot(static_cast<double>(offset.x), static_cast<double>(offset.y));
    cost += huberLoss(distance, config_.yaw_cost_huber_px);

    // 形状项：这条边（当前角点指向下一个角点）观测方向与投影方向的夹角。
    // 角点顺序是闭合四边形，所以取模就是四条边。
    const std::size_t next = (index + 1) % count;
    const cv::Point2d observed_edge{
      static_cast<double>(armor.points[next].x - armor.points[index].x),
      static_cast<double>(armor.points[next].y - armor.points[index].y)};
    const cv::Point2d projected_edge{
      static_cast<double>(projected[next].x - projected[index].x),
      static_cast<double>(projected[next].y - projected[index].y)};
    const double angle_deg =
      vectorAngle(observed_edge, projected_edge) * 180.0 / std::numbers::pi;
    cost += config_.yaw_cost_shape_weight * huberLoss(angle_deg, config_.yaw_cost_huber_deg);
  }

  return cost;
}

void PnpSolver::optimize_yaw(Armor & armor) const
{
  // 单板 PnP 的 yaw 在正视时极不稳定（板宽方向的透视变化太小），但装甲板的
  // 倾角是已知的，于是把 yaw 当成唯一自由量，在枪管朝向附近离散搜索重投影
  // 代价最小的角度。搜索窗口以枪管 yaw 为中心——这也是要求 barrel 系 x 轴
  // 必须指向枪口方向的原因。
  const double barrel_yaw = L6Telemetry::eulers(R_barrel2world_, 2, 1, 0)[0];

  // 搜索窗口的硬边界。细扫是在粗扫最优点附近展开的，落在窗口边上时会越界，
  // 所以两段扫描都夹在这里——窗口是设计约束，不能被细扫偷偷推出去。
  const double window_low = barrel_yaw - config_.yaw_search_range / 2.0;
  const double window_high = barrel_yaw + config_.yaw_search_range / 2.0;

  // 逐点扫描并记录最优值。yaw 用 limit_rad 归一化后再送进代价函数，但比较和
  // 细扫中心用的是**未归一化**的连续值，否则细扫窗口跨过 ±pi 时会被折断。
  const auto scan = [this, &armor, window_low, window_high](
                      double center, double half_range, double step, double & best_yaw,
                      double & best_cost) {
    if (!(step > 0.0) || !(half_range > 0.0)) {
      return;
    }
    const int steps = static_cast<int>(std::lround(2.0 * half_range / step));
    for (int index = 0; index <= steps; ++index) {
      const double yaw =
        std::clamp(center - half_range + index * step, window_low, window_high);
      const double cost = armor_reprojection_error(armor, L6Telemetry::limit_rad(yaw));
      if (cost < best_cost) {
        best_cost = cost;
        best_yaw = yaw;
      }
    }
  };

  double best_cost = std::numeric_limits<double>::infinity();
  double best_yaw = armor.ypr_in_world[0];

  // 粗扫：整个搜索窗口，步长大，只负责挑出代价盆地。
  scan(barrel_yaw, config_.yaw_search_range / 2.0, config_.yaw_coarse_step, best_yaw, best_cost);

  // 粗扫一个点都没算成（重投影全失败）时保持 IPPE 的原始 yaw，不要拿正无穷
  // 对应的角度去细扫。
  if (std::isfinite(best_cost)) {
    // 细扫：在粗扫最优点附近，步长小，负责精度。窗口至少覆盖一个粗扫步长，
    // 保证粗扫格点之间的空隙不会漏掉。
    const double fine_range = std::max(config_.yaw_fine_range, config_.yaw_coarse_step);
    scan(best_yaw, fine_range, config_.yaw_fine_step, best_yaw, best_cost);
  }

  armor.yaw_raw = armor.ypr_raw_in_world[0];
  armor.ypr_in_world[0] = L6Telemetry::limit_rad(best_yaw);
}

std::vector<cv::Point2f> PnpSolver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  if (!ready_ || !world_barrel_ready_) {
    return {};
  }

  const Eigen::Matrix3d R_armor2camera = R_camera2barrel_.transpose() *
                                         R_barrel2world_.transpose() *
                                         armorRotationInWorld(yaw, name);
  const Eigen::Vector3d t_armor2camera =
    R_camera2barrel_.transpose() * (R_barrel2world_.transpose() * xyz_in_world - t_camera2barrel_);

  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Vec3d rvec;
  const cv::Vec3d tvec{t_armor2camera.x(), t_armor2camera.y(), t_armor2camera.z()};
  std::vector<cv::Point2f> image_points;
  try {
    cv::Rodrigues(R_armor2camera_cv, rvec);
    cv::projectPoints(
      type == ArmorType::Big ? big_armor_points_ : small_armor_points_, rvec, tvec,
      calibration_.camera_matrix, calibration_.distortion_coefficients, image_points);
  } catch (const cv::Exception & error) {
    L6Telemetry::logWarn("PnpSolver armor reprojection failed", error.what());
    return {};
  }
  return image_points;
}

}  // namespace L3Estimation
