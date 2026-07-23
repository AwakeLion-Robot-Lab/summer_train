#include "l3_estimation/pnp_solver.hpp"

#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace L3Estimation {

PnpSolver::PnpSolver(
  L1Sensor::CameraCalibration calibration,
  ArmorDimensions dimensions)
  : calibration_(std::move(calibration)), dimensions_(dimensions)
{
}

std::array<cv::Point3f, 4> PnpSolver::objectPoints(ArmorSize size) const
{
  const float width = static_cast<float>(size == ArmorSize::Large
                         ? dimensions_.large_width
                         : dimensions_.small_width);
  const float half_width = width / 2.0F;
  const float half_height = static_cast<float>(dimensions_.height / 2.0);

  // 装甲板局部坐标：x 沿法向、y 向左、z 向上。
  return {{{0.0F,  half_width,  half_height},   // 左上
           {0.0F, -half_width,  half_height},   // 右上
           {0.0F, -half_width, -half_height},   // 右下
           {0.0F,  half_width, -half_height}}}; // 左下
}

std::optional<ArmorPose> PnpSolver::solve(
  const L2Perception::ArmorDetection& detection,
  ArmorSize size) const
{
  const std::vector<cv::Point2f> contour(
    detection.corners.begin(), detection.corners.end());
  if (std::abs(cv::contourArea(contour)) < 1.0) {
    return std::nullopt;
  }

  cv::Vec3d rvec;
  cv::Vec3d tvec;
  const bool success = cv::solvePnP(
    objectPoints(size), detection.corners,
    calibration_.camera_matrix, calibration_.distortion_coefficients,
    rvec, tvec, false, cv::SOLVEPNP_IPPE);

  if (!success || tvec[2] <= 0.0) {
    return std::nullopt;
  }

  return ArmorPose{.rvec = rvec, .tvec = tvec};
}

}  // namespace L3Estimation
