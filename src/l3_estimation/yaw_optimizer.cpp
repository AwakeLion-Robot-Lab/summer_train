#include "l3_estimation/yaw_optimizer.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace L3Estimation {
namespace {

double normalizeAngle(double angle) noexcept
{
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  angle = std::remainder(angle, kTwoPi);
  return angle <= -std::numbers::pi ? angle + kTwoPi : angle;
}

bool isRotationMatrix(const Eigen::Matrix3d& rotation) noexcept
{
  if (!rotation.allFinite()) {
    return false;
  }
  const Eigen::Matrix3d error =
    rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
  return error.norm() < 1e-5
         && std::abs(rotation.determinant() - 1.0) < 1e-5;
}

Eigen::Matrix3d rotationFromRvec(const cv::Vec3d& rvec)
{
  cv::Matx33d rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);
  Eigen::Matrix3d rotation;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      rotation(row, column) = rotation_cv(row, column);
    }
  }
  return rotation;
}

cv::Matx33d cvRotation(const Eigen::Matrix3d& rotation)
{
  cv::Matx33d result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result(row, column) = rotation(row, column);
    }
  }
  return result;
}

Eigen::Vector3d rotationToRpy(const Eigen::Matrix3d& rotation)
{
  const Eigen::Vector3d ypr = rotation.eulerAngles(2, 1, 0);
  return {ypr.z(), ypr.y(), ypr.x()};
}

}  // namespace

YawOptimizer::YawOptimizer(
  L1Sensor::CameraCalibration calibration,
  ArmorDimensions dimensions,
  YawSearchConfig config)
  : calibration_(std::move(calibration)),
    dimensions_(dimensions),
    config_(config)
{
  if (!calibration_.T_barrel_camera) {
    throw std::invalid_argument(
      "YawOptimizer requires calibrated T_barrel_camera");
  }
  R_barrel_camera_ = calibration_.T_barrel_camera->linear();
  if (!isRotationMatrix(R_barrel_camera_)
      || !std::isfinite(config_.search_half_range_rad)
      || config_.search_half_range_rad <= 0.0
      || config_.search_half_range_rad > std::numbers::pi
      || !std::isfinite(config_.search_step_rad)
      || config_.search_step_rad <= 0.0) {
    throw std::invalid_argument(
      "YawOptimizer received an invalid configuration");
  }
}

std::array<cv::Point3d, 4> YawOptimizer::objectPoints(
  ArmorSize size) const
{
  const double width = size == ArmorSize::Large
                         ? dimensions_.large_width
                         : dimensions_.small_width;
  const double half_width = width / 2.0;
  const double half_height = dimensions_.height / 2.0;
  return {{{0.0,  half_width,  half_height},
           {0.0, -half_width,  half_height},
           {0.0, -half_width, -half_height},
           {0.0,  half_width, -half_height}}};
}

std::optional<YawOptimizationResult> YawOptimizer::optimize(
  const L2Perception::ArmorDetection& detection,
  ArmorSize size,
  const ArmorPose& pose,
  const Eigen::Quaterniond& R_world_barrel,
  double configured_armor_pitch_rad) const
{
  if (!R_world_barrel.coeffs().allFinite()
      || R_world_barrel.norm() <= 1e-9
      || !std::isfinite(configured_armor_pitch_rad)) {
    return std::nullopt;
  }

  const Eigen::Matrix3d R_world_barrel_matrix =
    R_world_barrel.normalized().toRotationMatrix();
  if (!isRotationMatrix(R_world_barrel_matrix)) {
    return std::nullopt;
  }

  const Eigen::Matrix3d R_camera_armor =
    rotationFromRvec(pose.rvec);
  const Eigen::Matrix3d R_world_armor_raw =
    R_world_barrel_matrix * R_barrel_camera_ * R_camera_armor;
  const double yaw_raw_world = normalizeAngle(std::atan2(
    R_world_armor_raw(1, 0),
    R_world_armor_raw(0, 0)));
  const double barrel_yaw_world = normalizeAngle(std::atan2(
    R_world_barrel_matrix(1, 0),
    R_world_barrel_matrix(0, 0)));

  const Eigen::Matrix3d R_camera_world =
    (R_world_barrel_matrix * R_barrel_camera_).transpose();
  const auto object_points = objectPoints(size);

  // 给定一个世界系 yaw，固定 pitch 和 PnP 位置后计算四角点二维 RMSE。
  const auto reprojection_error =
    [&](double yaw) -> std::optional<double> {
    const Eigen::Matrix3d R_world_armor =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix()
      * Eigen::AngleAxisd(
          configured_armor_pitch_rad,
          Eigen::Vector3d::UnitY()).toRotationMatrix();
    const Eigen::Matrix3d R_camera_armor_constrained =
      R_camera_world * R_world_armor;
    if (!isRotationMatrix(R_camera_armor_constrained)) {
      return std::nullopt;
    }

    for (const auto& point : object_points) {
      const Eigen::Vector3d point_camera =
        R_camera_armor_constrained
          * Eigen::Vector3d{point.x, point.y, point.z}
        + Eigen::Vector3d{pose.tvec[0], pose.tvec[1], pose.tvec[2]};
      if (!point_camera.allFinite() || point_camera.z() <= 0.0) {
        return std::nullopt;
      }
    }

    cv::Vec3d constrained_rvec;
    cv::Rodrigues(
      cvRotation(R_camera_armor_constrained),
      constrained_rvec);
    std::vector<cv::Point2d> projected;
    cv::projectPoints(
      object_points,
      constrained_rvec,
      pose.tvec,
      calibration_.camera_matrix,
      calibration_.distortion_coefficients,
      projected);
    if (projected.size() != detection.corners.size()) {
      return std::nullopt;
    }

    double squared_error_sum = 0.0;
    for (std::size_t index = 0; index < projected.size(); ++index) {
      const double dx =
        projected[index].x - detection.corners[index].x;
      const double dy =
        projected[index].y - detection.corners[index].y;
      if (!std::isfinite(dx) || !std::isfinite(dy)) {
        return std::nullopt;
      }
      squared_error_sum += dx * dx + dy * dy;
    }
    return std::sqrt(
      squared_error_sum / static_cast<double>(projected.size()));
  };

  const auto raw_error = reprojection_error(yaw_raw_world);
  if (!raw_error) {
    return std::nullopt;
  }

  double best_yaw = yaw_raw_world;
  double best_error = *raw_error;
  int evaluated_yaw_count = 1;
  if (config_.enabled) {
    const int half_step_count = static_cast<int>(
      std::floor(
        config_.search_half_range_rad / config_.search_step_rad));
    for (int step = -half_step_count;
         step <= half_step_count;
         ++step) {
      const double yaw = normalizeAngle(
        barrel_yaw_world
        + static_cast<double>(step) * config_.search_step_rad);
      const auto error = reprojection_error(yaw);
      if (!error) {
        continue;
      }
      ++evaluated_yaw_count;
      if (*error < best_error) {
        best_error = *error;
        best_yaw = yaw;
      }
    }
  }

  // 与 tongjiceshi 基线一致：重投影误差用于选 yaw 和诊断，
  // 不在这一层拒绝已经通过 PnP 检查的观测。
  if (!std::isfinite(best_error)) {
    return std::nullopt;
  }
  return YawOptimizationResult{
    .rpy_raw_world = rotationToRpy(R_world_armor_raw),
    .yaw_raw_world = yaw_raw_world,
    .yaw_optimized_world = normalizeAngle(best_yaw),
    .pnp_reprojection_error_px = pose.reprojection_error_px,
    .raw_yaw_reprojection_error_px = *raw_error,
    .optimized_reprojection_error_px = best_error,
    .evaluated_yaw_count = evaluated_yaw_count};
}

}  // namespace L3Estimation
