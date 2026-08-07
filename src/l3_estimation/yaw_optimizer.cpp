#include "l3_estimation/yaw_optimizer.hpp"
#include "l3_estimation/angle_utils.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace L3Estimation {
namespace {

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
  YawOptimizationConfig config)
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
      || config_.max_iterations <= 0
      || !std::isfinite(config_.parameter_step)
      || config_.parameter_step <= 0.0
      || !std::isfinite(config_.convergence_tolerance)
      || config_.convergence_tolerance <= 0.0) {
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

std::optional<YawOptimizationResult> YawOptimizer::raw(
  const L2Perception::ArmorDetection& detection,
  ArmorSize size,
  const ArmorPose& pose,
  const Eigen::Quaterniond& R_world_barrel,
  double configured_armor_pitch_rad) const
{
  // 只返回 raw 评估结果（不做 yaw 搜索）。
  return evaluateRaw(
    detection,
    size,
    pose,
    R_world_barrel,
    configured_armor_pitch_rad);
}

std::optional<YawOptimizationResult> YawOptimizer::evaluateRaw(
  const L2Perception::ArmorDetection& detection,
  ArmorSize size,
  const ArmorPose& pose,
  const Eigen::Quaterniond& R_world_barrel,
  double configured_armor_pitch_rad) const
{
  // 校验输入，计算世界系 raw yaw 与固定 pitch 模型的 raw 重投影误差。
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
  const Eigen::Matrix3d R_camera_world =
    (R_world_barrel_matrix * R_barrel_camera_).transpose();
  const auto object_points = objectPoints(size);

  const Eigen::Vector4d theta_raw{
    pose.tvec[0],
    pose.tvec[1],
    pose.tvec[2],
    yaw_raw_world};
  const auto raw_residual = reprojectionResidual(
    theta_raw,
    configured_armor_pitch_rad,
    R_camera_world,
    object_points,
    detection);
  if (!raw_residual) {
    return std::nullopt;
  }
  const double raw_error = residualRmse(*raw_residual);

  return YawOptimizationResult{
    .rpy_raw_world = rotationToRpy(R_world_armor_raw),
    .yaw_raw_world = yaw_raw_world,
    .yaw_optimized_world = normalizeAngle(yaw_raw_world),
    .tvec_optimized_camera = pose.tvec,
    .pnp_reprojection_error_px = pose.reprojection_error_px,
    .raw_yaw_reprojection_error_px = raw_error,
    .optimized_reprojection_error_px = raw_error,
    .evaluation_count = 1};
}

std::optional<YawOptimizer::ResidualVector>
YawOptimizer::reprojectionResidual(
  const Eigen::Vector4d& theta,
  double configured_armor_pitch_rad,
  const Eigen::Matrix3d& R_camera_world,
  const std::array<cv::Point3d, 4>& object_points,
  const L2Perception::ArmorDetection& detection) const
{
  // 固定 pitch 模型下投影四个角点并求 8 维像素残差；投影非法返回 nullopt。
  const Eigen::Vector3d tvec{theta[0], theta[1], theta[2]};
  const Eigen::Matrix3d R_world_armor =
    Eigen::AngleAxisd(theta[3], Eigen::Vector3d::UnitZ()).toRotationMatrix()
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
      + tvec;
    if (!point_camera.allFinite() || point_camera.z() <= 0.0) {
      return std::nullopt;
    }
  }

  cv::Vec3d constrained_rvec;
  cv::Rodrigues(
    cvRotation(R_camera_armor_constrained),
    constrained_rvec);
  const cv::Vec3d tvec_cv{tvec[0], tvec[1], tvec[2]};
  std::vector<cv::Point2d> projected;
  cv::projectPoints(
    object_points,
    constrained_rvec,
    tvec_cv,
    calibration_.camera_matrix,
    calibration_.distortion_coefficients,
    projected);
  if (projected.size() != detection.corners.size()) {
    return std::nullopt;
  }

  ResidualVector residual = ResidualVector::Zero();
  for (std::size_t index = 0; index < projected.size(); ++index) {
    const double dx =
      projected[index].x - detection.corners[index].x;
    const double dy =
      projected[index].y - detection.corners[index].y;
    if (!std::isfinite(dx) || !std::isfinite(dy)) {
      return std::nullopt;
    }
    residual[2 * index] = dx;
    residual[2 * index + 1] = dy;
  }
  return residual;
}

double YawOptimizer::residualRmse(
  const ResidualVector& residual) noexcept
{
  return std::sqrt(residual.squaredNorm() / 8.0);
}

std::optional<YawOptimizationResult> YawOptimizer::optimize(
  const L2Perception::ArmorDetection& detection,
  ArmorSize size,
  const ArmorPose& pose,
  const Eigen::Quaterniond& R_world_barrel,
  double configured_armor_pitch_rad) const
{
  // 连续 Gauss-Newton：θ = [yaw]（PnP 位置与模型 pitch 固定），初值为
  // raw yaw。每一步只在改进残差平方和时接受；优化失败或未改进时保留
  // raw 结果，不丢弃已经通过 PnP 检查的观测。
  const auto raw_result = evaluateRaw(
    detection,
    size,
    pose,
    R_world_barrel,
    configured_armor_pitch_rad);
  if (!raw_result) {
    return std::nullopt;
  }
  if (!config_.enabled) {
    return raw_result;
  }

  const Eigen::Matrix3d R_world_barrel_matrix =
    R_world_barrel.normalized().toRotationMatrix();
  const Eigen::Matrix3d R_camera_world =
    (R_world_barrel_matrix * R_barrel_camera_).transpose();
  const auto object_points = objectPoints(size);

  Eigen::Vector4d theta{
    pose.tvec[0],
    pose.tvec[1],
    pose.tvec[2],
    raw_result->yaw_raw_world};
  auto residual = reprojectionResidual(
    theta,
    configured_armor_pitch_rad,
    R_camera_world,
    object_points,
    detection);
  if (!residual) {
    return raw_result;
  }
  int evaluation_count = 1;

  // 连续 Gauss-Newton：θ = [yaw]，PnP 位置保持固定。
  double cost = residual->squaredNorm();
  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    Eigen::Vector4d perturbed = theta;
    perturbed[3] += config_.parameter_step;
    const auto perturbed_residual = reprojectionResidual(
      perturbed,
      configured_armor_pitch_rad,
      R_camera_world,
      object_points,
      detection);
    ++evaluation_count;
    if (!perturbed_residual) {
      break;
    }
    const Eigen::Matrix<double, 8, 1> jacobian =
      (*perturbed_residual - *residual) / config_.parameter_step;
    const double normal = jacobian.squaredNorm();
    const double gradient = jacobian.dot(*residual);
    if (!std::isfinite(normal) || normal <= 0.0
        || !std::isfinite(gradient)) {
      break;
    }
    const double step = -gradient / normal;

    Eigen::Vector4d candidate = theta;
    candidate[3] += step;
    const auto candidate_residual = reprojectionResidual(
      candidate,
      configured_armor_pitch_rad,
      R_camera_world,
      object_points,
      detection);
    ++evaluation_count;
    if (!candidate_residual) {
      break;
    }
    const double candidate_cost = candidate_residual->squaredNorm();
    if (!(candidate_cost < cost)) {
      break;
    }
    theta = candidate;
    residual = candidate_residual;
    cost = candidate_cost;
    if (std::abs(step) < config_.convergence_tolerance) {
      break;
    }
  }

  const double optimized_error = residualRmse(*residual);
  if (!std::isfinite(optimized_error)) {
    return raw_result;
  }
  return YawOptimizationResult{
    .rpy_raw_world = raw_result->rpy_raw_world,
    .yaw_raw_world = raw_result->yaw_raw_world,
    .yaw_optimized_world = normalizeAngle(theta[3]),
    .tvec_optimized_camera = {theta[0], theta[1], theta[2]},
    .pnp_reprojection_error_px = raw_result->pnp_reprojection_error_px,
    .raw_yaw_reprojection_error_px =
      raw_result->raw_yaw_reprojection_error_px,
    .optimized_reprojection_error_px = optimized_error,
    .evaluation_count = evaluation_count};
}

}  // namespace L3Estimation
