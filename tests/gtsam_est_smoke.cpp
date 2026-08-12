#include "l3_estimation/gtsam_est/factors.hpp"
#include "l3_estimation/gtsam_est/target.hpp"
#include "l6_telemetry/math.hpp"

#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/linear/NoiseModel.h>

#include <opencv2/core.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using L3Estimation::GtsamEst::ArmorRadiusCenterZFactor;
using L3Estimation::GtsamEst::ArmorRadiusDZFactor;
using L3Estimation::GtsamEst::ArmorReprojFactor;
using L3Estimation::GtsamEst::Target;

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void requireNear(
  const gtsam::Matrix& actual,
  const gtsam::Matrix& expected,
  double tolerance,
  const std::string& label)
{
  require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
    label + " shape mismatch");
  const double error = (actual - expected).cwiseAbs().maxCoeff();
  if (!(error <= tolerance)) {
    throw std::runtime_error(label + " max error = " + std::to_string(error));
  }
}

void testGeometryJacobians()
{
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(4, 1.0);
  Eigen::Isometry3d T_world_camera = Eigen::Isometry3d::Identity();
  T_world_camera.linear() =
    L6Telemetry::yprToRotation(Eigen::Vector3d{0.23, -0.08, 0.04});
  T_world_camera.translation() = Eigen::Vector3d{0.1, -0.05, 0.03};

  ArmorRadiusCenterZFactor factor(
    noise, 1, 2, 3, 4, T_world_camera, 2, 0.05, 0.5, 4);
  const gtsam::Pose3 pose{
    gtsam::Rot3{L6Telemetry::yprToRotation(Eigen::Vector3d{-0.31, 0.12, -0.07})},
    gtsam::Point3{2.1, 0.4, 0.6}};
  const double raw_radius = -0.45;
  const gtsam::Rot2 center_yaw = gtsam::Rot2::fromAngle(0.37);
  const gtsam::Point3 center{2.3, 0.7, 0.72};

  gtsam::Matrix H1, H2, H3, H4;
  factor.evaluateError(pose, raw_radius, center_yaw, center, &H1, &H2, &H3, &H4);
  const auto error = [&factor](
                       const gtsam::Pose3& p,
                       const double& radius,
                       const gtsam::Rot2& yaw,
                       const gtsam::Point3& point) {
    return factor.evaluateError(p, radius, yaw, point, nullptr, nullptr, nullptr, nullptr);
  };

  requireNear(H1, gtsam::numericalDerivative41(error, pose, raw_radius, center_yaw, center),
    2e-5, "ArmorRadiusCenterZFactor H1");
  requireNear(H2, gtsam::numericalDerivative42(error, pose, raw_radius, center_yaw, center),
    2e-6, "ArmorRadiusCenterZFactor H2");
  requireNear(H3, gtsam::numericalDerivative43(error, pose, raw_radius, center_yaw, center),
    2e-6, "ArmorRadiusCenterZFactor H3");
  requireNear(H4, gtsam::numericalDerivative44(error, pose, raw_radius, center_yaw, center),
    2e-6, "ArmorRadiusCenterZFactor H4");
}

void testAlternateGeometryJacobians()
{
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(4, 1.0);
  Eigen::Isometry3d T_world_camera = Eigen::Isometry3d::Identity();
  T_world_camera.linear() =
    L6Telemetry::yprToRotation(Eigen::Vector3d{-0.18, 0.07, -0.03});
  T_world_camera.translation() = Eigen::Vector3d{-0.08, 0.04, 0.02};

  ArmorRadiusDZFactor factor(
    noise, 1, 2, 3, 4, 5, T_world_camera, 1, 0.05, 0.5, 4);
  const gtsam::Pose3 pose{
    gtsam::Rot3{L6Telemetry::yprToRotation(Eigen::Vector3d{0.28, -0.09, 0.06})},
    gtsam::Point3{2.4, -0.3, 0.52}};
  const double raw_radius = 0.34;
  const double dz = -0.025;
  const gtsam::Rot2 center_yaw = gtsam::Rot2::fromAngle(-0.22);
  const gtsam::Point3 center{2.15, -0.1, 0.57};

  gtsam::Matrix H1, H2, H3, H4, H5;
  factor.evaluateError(
    pose, raw_radius, dz, center_yaw, center, &H1, &H2, &H3, &H4, &H5);
  const auto error = [&factor](
                       const gtsam::Pose3& p,
                       const double& radius,
                       const double& height,
                       const gtsam::Rot2& yaw,
                       const gtsam::Point3& point) {
    return factor.evaluateError(
      p, radius, height, yaw, point, nullptr, nullptr, nullptr, nullptr, nullptr);
  };

  requireNear(
    H1,
    gtsam::numericalDerivative51(error, pose, raw_radius, dz, center_yaw, center),
    2e-5, "ArmorRadiusDZFactor H1");
  requireNear(
    H2,
    gtsam::numericalDerivative52(error, pose, raw_radius, dz, center_yaw, center),
    2e-6, "ArmorRadiusDZFactor H2");
  requireNear(
    H3,
    gtsam::numericalDerivative53(error, pose, raw_radius, dz, center_yaw, center),
    2e-6, "ArmorRadiusDZFactor H3");
  requireNear(
    H4,
    gtsam::numericalDerivative54(error, pose, raw_radius, dz, center_yaw, center),
    2e-6, "ArmorRadiusDZFactor H4");
  requireNear(
    H5,
    gtsam::numericalDerivative55(error, pose, raw_radius, dz, center_yaw, center),
    2e-6, "ArmorRadiusDZFactor H5");
}

L1Sensor::CameraCalibration makeCalibration()
{
  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {1280, 1024};
  calibration.camera_matrix =
    (cv::Mat_<double>(3, 3) << 1000.0, 0.0, 640.0, 0.0, 1000.0, 512.0, 0.0, 0.0, 1.0);
  calibration.distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F);
  calibration.T_barrel_camera = Eigen::Isometry3d::Identity();
  return calibration;
}

void testReprojectionJacobian()
{
  const L1Sensor::CameraCalibration calibration = makeCalibration();
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(2, 1.0);
  const gtsam::Pose3 pose{
    gtsam::Rot3{L6Telemetry::yprToRotation(Eigen::Vector3d{0.2, -0.1, 0.04})},
    gtsam::Point3{0.08, -0.03, 3.2}};

  ArmorReprojFactor factor(
    noise, 1, calibration.camera_matrix, calibration.distortion_coefficients,
    L3Estimation::ArmorType::Small, {}, 0, Eigen::Vector2d{640.0, 512.0});
  gtsam::Matrix H;
  factor.evaluateError(pose, &H);
  const auto error = [&factor](const gtsam::Pose3& value) {
    return factor.evaluateError(value, nullptr);
  };
  requireNear(
    H, gtsam::numericalDerivative11<gtsam::Vector, gtsam::Pose3>(error, pose),
    2e-5, "ArmorReprojFactor H");
}

L3Estimation::Armor makeObservation(double elapsed_seconds)
{
  constexpr double center_start = 4.0;
  constexpr double velocity = 0.25;
  constexpr double radius = 0.2;
  L3Estimation::Armor armor;
  armor.name = L3Estimation::ArmorName::Infantry3;
  armor.type = L3Estimation::ArmorType::Small;
  armor.xyz_in_world = {
    center_start + velocity * elapsed_seconds - radius, 0.0, 0.45};
  armor.xyz_in_camera = armor.xyz_in_world;
  armor.ypr_in_world.setZero();
  armor.ypr_raw_in_world.setZero();
  armor.ypr_in_camera.setZero();
  armor.ypd_in_world = L6Telemetry::xyz2ypd(armor.xyz_in_world);

  const L3Estimation::ArmorConfig geometry;
  const std::array<Eigen::Vector3d, 4> corners{
    Eigen::Vector3d{0.0, geometry.small_width * 0.5, geometry.height * 0.5},
    Eigen::Vector3d{0.0, -geometry.small_width * 0.5, geometry.height * 0.5},
    Eigen::Vector3d{0.0, -geometry.small_width * 0.5, -geometry.height * 0.5},
    Eigen::Vector3d{0.0, geometry.small_width * 0.5, -geometry.height * 0.5}};
  constexpr double fx = 1000.0;
  constexpr double fy = 1000.0;
  constexpr double cx = 640.0;
  constexpr double cy = 512.0;
  const Eigen::Matrix3d rotation =
    L6Telemetry::yprToRotation(armor.ypr_in_camera);
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const Eigen::Vector3d point = rotation * corners[index] + armor.xyz_in_camera;
    armor.points[index] = cv::Point2f{
      static_cast<float>(fx * point.x() / point.z() + cx),
      static_cast<float>(fy * point.y() / point.z() + cy)};
  }
  return armor;
}

void testIsamTarget()
{
  L3Estimation::TargetConfig target_config;
  L3Estimation::GtsamEst::Config graph_config;
  graph_config.first_update_batch_size = 3;
  graph_config.default_radius = 0.2;
  graph_config.lost_threshold = std::chrono::milliseconds(1000);

  const L1Sensor::CameraCalibration calibration = makeCalibration();
  Target target(target_config, graph_config, {}, calibration);
  constexpr int frame_count = 120;
  constexpr int frame_interval_ms = 20;
  const L3Estimation::TimePoint start{};
  const Eigen::Isometry3d T_world_camera = Eigen::Isometry3d::Identity();

  L3Estimation::Armor observation = makeObservation(0.0);
  target.initialize(observation, start, T_world_camera);
  for (int frame = 1; frame < frame_count; ++frame) {
    const double elapsed = frame * frame_interval_ms / 1000.0;
    observation = makeObservation(elapsed);
    const std::vector<const L3Estimation::Armor*> observations{&observation};
    const auto timestamp = start + std::chrono::milliseconds(frame * frame_interval_ms);
    require(target.update(observations, timestamp, T_world_camera),
      "synthetic armor should stay associated");
  }

  const L3Estimation::TrackedTarget snapshot = target.snapshot();
  if (!snapshot.valid()) {
    std::cerr << "state: " << snapshot.state().transpose() << '\n';
    std::cerr << "covariance diagonal: " << snapshot.covariance().diagonal().transpose() << '\n';
    std::cerr << "armor count: " << snapshot.armorCount() << '\n';
  }
  require(snapshot.valid(), "GTSAM snapshot must be finite");
  require(!target.diverged(), "GTSAM target must not diverge on exact observations");
  require(std::abs(snapshot.state()[L3Estimation::RadiusA] - 0.2) < 0.03,
    "radius estimate drifted away from the synthetic geometry");
  require(snapshot.state()[L3Estimation::VelocityX] > 0.05,
    "factor graph did not recover positive center velocity");
  require(target.activeVariableCount() > static_cast<std::size_t>(frame_count * 4),
    "ISAM2 did not retain the complete JLU tracking graph");
}

}  // namespace

int main()
{
  try {
    testGeometryJacobians();
    testAlternateGeometryJacobians();
    testReprojectionJacobian();
    testIsamTarget();
  } catch (const std::exception& exception) {
    std::cerr << "gtsam_est_smoke failed: " << exception.what() << '\n';
    return 1;
  }
  std::cout << "gtsam_est_smoke passed\n";
  return 0;
}
