#include "l3_estimation/pnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace L3Estimation {
namespace {

bool finitePoint(const cv::Point2f& point) noexcept
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool finiteVector(const cv::Vec3d& vector) noexcept
{
  return std::isfinite(vector[0])
         && std::isfinite(vector[1])
         && std::isfinite(vector[2]);
}

std::optional<cv::Vec3d> toVec3d(const cv::Mat& mat) noexcept
{
  if (mat.total() != 3 || mat.type() != CV_64FC1) {
    return std::nullopt;
  }
  const cv::Mat column = mat.reshape(1, 3);
  return cv::Vec3d{
    column.at<double>(0),
    column.at<double>(1),
    column.at<double>(2)};
}

}  // namespace

PnpSolver::PnpSolver(
  L1Sensor::CameraCalibration calibration,
  ArmorDimensions dimensions,
  PnpSolverConfig config)
  : calibration_(std::move(calibration)),
    dimensions_(dimensions),
    config_(config)
{
  const bool dimensions_valid =
    std::isfinite(dimensions_.small_width)
    && dimensions_.small_width > 0.0
    && std::isfinite(dimensions_.large_width)
    && dimensions_.large_width > 0.0
    && std::isfinite(dimensions_.height)
    && dimensions_.height > 0.0;
  const bool config_valid =
    std::isfinite(config_.minimum_corner_area_px)
    && config_.minimum_corner_area_px > 0.0
    && std::isfinite(config_.minimum_distance_m)
    && config_.minimum_distance_m >= 0.0
    && std::isfinite(config_.maximum_distance_m)
    && config_.maximum_distance_m > config_.minimum_distance_m
    && std::isfinite(config_.maximum_reprojection_error_px)
    && config_.maximum_reprojection_error_px > 0.0;
  if (!dimensions_valid || !config_valid) {
    throw std::invalid_argument("PnpSolver received an invalid configuration");
  }
  if (calibration_.camera_matrix.rows != 3
      || calibration_.camera_matrix.cols != 3
      || !cv::checkRange(calibration_.camera_matrix)
      || !cv::checkRange(calibration_.distortion_coefficients)) {
    throw std::invalid_argument("PnpSolver received an invalid calibration");
  }
}

std::array<cv::Point3f, 4> PnpSolver::objectPoints(
  ArmorSize size) const
{
  const float width = static_cast<float>(
    size == ArmorSize::Large
      ? dimensions_.large_width
      : dimensions_.small_width);
  const float half_width = width / 2.0F;
  const float half_height =
    static_cast<float>(dimensions_.height / 2.0);

  // 装甲局部系：x 沿法向，y 向左，z 向上。
  return {{{0.0F,  half_width,  half_height},
           {0.0F, -half_width,  half_height},
           {0.0F, -half_width, -half_height},
           {0.0F,  half_width, -half_height}}};
}

std::vector<ArmorPose> PnpSolver::solve(
  const L2Perception::ArmorDetection& detection,
  ArmorSize size) const
{
  // 先拒绝非有限、非凸或面积太小的四边形。
  if (!std::all_of(
        detection.corners.begin(),
        detection.corners.end(),
        finitePoint)) {
    return {};
  }
  const std::vector<cv::Point2f> contour(
    detection.corners.begin(),
    detection.corners.end());
  if (!cv::isContourConvex(contour)
      || std::abs(cv::contourArea(contour))
           < config_.minimum_corner_area_px) {
    return {};
  }

  const auto object_points = objectPoints(size);
  std::vector<ArmorPose> poses;

  if (config_.enable_ippe_dual_candidates) {
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    const int count = cv::solvePnPGeneric(
      object_points,
      detection.corners,
      calibration_.camera_matrix,
      calibration_.distortion_coefficients,
      rvecs,
      tvecs,
      false,
      cv::SOLVEPNP_IPPE);
    const int candidate_count = std::min(count, 2);
    for (int index = 0; index < candidate_count; ++index) {
      const auto rvec =
        toVec3d(rvecs[static_cast<std::size_t>(index)]);
      const auto tvec =
        toVec3d(tvecs[static_cast<std::size_t>(index)]);
      if (!rvec || !tvec) {
        continue;
      }
      if (auto pose = makeCandidate(
            index,
            *rvec,
            *tvec,
            detection,
            object_points)) {
        poses.push_back(*pose);
      }
    }
  } else {
    // 关闭开关时恢复旧单候选路径，保证与升级前行为严格一致。
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    const bool solved = cv::solvePnP(
      object_points,
      detection.corners,
      calibration_.camera_matrix,
      calibration_.distortion_coefficients,
      rvec,
      tvec,
      false,
      cv::SOLVEPNP_IPPE);
    if (solved) {
      if (auto pose = makeCandidate(
            0,
            rvec,
            tvec,
            detection,
            object_points)) {
        poses.push_back(*pose);
      }
    }
  }
  return poses;
}

std::optional<ArmorPose> PnpSolver::makeCandidate(
  int candidate_index,
  const cv::Vec3d& rvec,
  const cv::Vec3d& tvec,
  const L2Perception::ArmorDetection& detection,
  const std::array<cv::Point3f, 4>& object_points) const
{
  // 对单个候选依次做有限性、正深度、距离与四角点 RMSE 检查。
  if (!finiteVector(rvec) || !finiteVector(tvec)) {
    return std::nullopt;
  }

  const double distance = cv::norm(tvec);
  if (tvec[2] <= 0.0
      || distance < config_.minimum_distance_m
      || distance > config_.maximum_distance_m) {
    return std::nullopt;
  }

  // 平面中心在相机前方还不够，四个物点也必须全部保持正深度。
  cv::Matx33d rotation;
  cv::Rodrigues(rvec, rotation);
  for (const auto& point : object_points) {
    const cv::Vec3d point_camera =
      rotation * cv::Vec3d{point.x, point.y, point.z} + tvec;
    if (!finiteVector(point_camera) || point_camera[2] <= 0.0) {
      return std::nullopt;
    }
  }

  // 用四个角点的二维 RMSE 作为当前单解的质量指标。
  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    object_points,
    rvec,
    tvec,
    calibration_.camera_matrix,
    calibration_.distortion_coefficients,
    projected);
  double squared_error_sum = 0.0;
  for (std::size_t index = 0; index < projected.size(); ++index) {
    const cv::Point2f residual =
      detection.corners[index] - projected[index];
    squared_error_sum += residual.dot(residual);
  }
  const double reprojection_error =
    std::sqrt(squared_error_sum / static_cast<double>(projected.size()));
  if (!std::isfinite(reprojection_error)
      || reprojection_error > config_.maximum_reprojection_error_px) {
    return std::nullopt;
  }

  return ArmorPose{
    .rvec = rvec,
    .tvec = tvec,
    .reprojection_error_px = reprojection_error,
    .ippe_candidate_index = candidate_index};
}

}  // namespace L3Estimation
