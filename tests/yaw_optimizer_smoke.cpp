#include "l3_estimation/yaw_optimizer.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
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

double angleError(double lhs, double rhs)
{
  return std::abs(std::remainder(
    lhs - rhs,
    2.0 * std::numbers::pi));
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

  Eigen::Isometry3d T_barrel_camera =
    Eigen::Isometry3d::Identity();
  T_barrel_camera.linear() <<
     0.0,  0.0,  1.0,
    -1.0,  0.0,  0.0,
     0.0, -1.0,  0.0;
  calibration.T_barrel_camera = T_barrel_camera;
  return calibration;
}

L2Perception::ArmorDetection makeDetection(
  const L1Sensor::CameraCalibration& calibration,
  double yaw_world,
  double pitch,
  const cv::Vec3d& tvec,
  cv::Vec3d& rvec)
{
  const Eigen::Matrix3d R_world_armor =
    Eigen::AngleAxisd(
      yaw_world,
      Eigen::Vector3d::UnitZ()).toRotationMatrix()
    * Eigen::AngleAxisd(
      pitch,
      Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d R_camera_armor =
    calibration.T_barrel_camera->linear().transpose()
    * R_world_armor;
  cv::Rodrigues(cvRotation(R_camera_armor), rvec);

  const std::array<cv::Point3d, 4> object_points{{
    {0.0,  0.0675,  0.0275},
    {0.0, -0.0675,  0.0275},
    {0.0, -0.0675, -0.0275},
    {0.0,  0.0675, -0.0275}}};
  std::vector<cv::Point2d> projected;
  cv::projectPoints(
    object_points,
    rvec,
    tvec,
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    projected);

  L2Perception::ArmorDetection detection;
  for (std::size_t index = 0; index < projected.size(); ++index) {
    detection.corners[index] = {
      static_cast<float>(projected[index].x),
      static_cast<float>(projected[index].y)};
  }
  detection.class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry3);
  detection.confidence = 1.0F;
  return detection;
}

}  // namespace

int main()
{
  try {
    constexpr double kPitch =
      15.0 * std::numbers::pi / 180.0;
    constexpr double kExpectedYaw = 0.37;
    const cv::Vec3d tvec{0.06, -0.02, 3.0};
    const auto calibration = makeCalibration();
    cv::Vec3d exact_rvec;
    const auto detection = makeDetection(
      calibration,
      kExpectedYaw,
      kPitch,
      tvec,
      exact_rvec);

    L3Estimation::YawOptimizer optimizer{
      calibration,
      {},
      {}};

    // 故意把原始 yaw 旋转偏离真值，验证 1°遍历能找回附近最小值。
    const Eigen::Matrix3d R_world_armor_wrong =
      Eigen::AngleAxisd(
        kExpectedYaw + 0.20,
        Eigen::Vector3d::UnitZ()).toRotationMatrix()
      * Eigen::AngleAxisd(
        kPitch,
        Eigen::Vector3d::UnitY()).toRotationMatrix();
    cv::Vec3d wrong_rvec;
    cv::Rodrigues(
      cvRotation(
        calibration.T_barrel_camera->linear().transpose()
        * R_world_armor_wrong),
      wrong_rvec);
    const L3Estimation::ArmorPose pose{
      .rvec = wrong_rvec,
      .tvec = tvec,
      .reprojection_error_px = 0.0};

    const auto result = optimizer.optimize(
      detection,
      L3Estimation::ArmorSize::Small,
      pose,
      Eigen::Quaterniond::Identity(),
      kPitch);
    require(result.has_value(), "1-degree yaw search failed");
    require(
      result->rpy_raw_world.allFinite()
        && angleError(result->rpy_raw_world.x(), 0.0) < 1e-9
        && angleError(result->rpy_raw_world.y(), kPitch) < 1e-9
        && angleError(
             result->rpy_raw_world.z(),
             kExpectedYaw + 0.20) < 1e-9,
      "raw world RPY does not match the PnP pose");
    require(
      angleError(result->yaw_optimized_world, kExpectedYaw)
        <= 1.1 * std::numbers::pi / 180.0,
      "1-degree yaw search did not recover the known yaw");
    require(
      result->optimized_reprojection_error_px
        <= result->raw_yaw_reprojection_error_px,
      "yaw search made the constrained reprojection error worse");
    require(
      result->evaluated_yaw_count >= 140,
      "yaw search did not cover the configured range");
  } catch (const std::exception& error) {
    std::cerr << "YawOptimizer smoke test failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "YawOptimizer smoke test passed\n";
  return 0;
}
