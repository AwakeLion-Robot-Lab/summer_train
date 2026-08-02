#include "l3_estimation/pnp_solver.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace {

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

L1Sensor::CameraCalibration makeCalibration()
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {1280, 1024};
  calibration.camera_matrix = (cv::Mat_<double>(3, 3)
    << 1200.0, 0.0, 640.0,
       0.0, 1200.0, 512.0,
       0.0, 0.0, 1.0);
  calibration.distortion_coefficients =
    cv::Mat::zeros(1, 5, CV_64FC1);
  return calibration;
}

L2Perception::ArmorDetection makeDetection(
  const L1Sensor::CameraCalibration& calibration,
  const cv::Vec3d& rvec,
  const cv::Vec3d& tvec)
{
  const std::array<cv::Point3f, 4> object_points{{
    {0.0F,  0.0675F,  0.0275F},
    {0.0F, -0.0675F,  0.0275F},
    {0.0F, -0.0675F, -0.0275F},
    {0.0F,  0.0675F, -0.0275F}}};
  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    object_points,
    rvec,
    tvec,
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    projected);

  L2Perception::ArmorDetection detection;
  std::copy(
    projected.begin(),
    projected.end(),
    detection.corners.begin());
  detection.class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry3);
  detection.confidence = 1.0F;
  return detection;
}

}  // namespace

int main()
{
  try {
    const auto calibration = makeCalibration();
    L3Estimation::PnpSolver solver{calibration};
    const cv::Matx33d rotation_armor_to_camera{
       0.0, -1.0,  0.0,
       0.0,  0.0, -1.0,
       1.0,  0.0,  0.0};
    cv::Vec3d rvec;
    cv::Rodrigues(rotation_armor_to_camera, rvec);
    const cv::Vec3d expected_tvec{0.08, -0.03, 3.2};
    const auto detection = makeDetection(
      calibration,
      rvec,
      expected_tvec);

    // 基线只返回一个经过基本几何检查的 IPPE 解。
    const auto pose = solver.solve(
      detection,
      L3Estimation::ArmorSize::Small);
    require(pose.has_value(), "single IPPE solve failed");
    require(
      cv::norm(pose->tvec - expected_tvec) < 1e-3,
      "single IPPE translation is inaccurate");
    require(
      std::isfinite(pose->reprojection_error_px)
        && pose->reprojection_error_px < 1e-3,
      "single IPPE reprojection RMSE is invalid");

    auto degenerate = detection;
    degenerate.corners.fill({100.0F, 100.0F});
    require(
      !solver.solve(
        degenerate,
        L3Estimation::ArmorSize::Small),
      "degenerate quadrilateral was accepted");

    auto non_finite = detection;
    non_finite.corners[0].x =
      std::numeric_limits<float>::quiet_NaN();
    require(
      !solver.solve(
        non_finite,
        L3Estimation::ArmorSize::Small),
      "non-finite corner was accepted");
  } catch (const std::exception& error) {
    std::cerr << "PnpSolver smoke test failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "PnpSolver smoke test passed\n";
  return 0;
}
