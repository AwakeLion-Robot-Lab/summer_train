#include "l3_estimation/config.hpp"
#include "l3_estimation/target_estimator.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <optional>
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
  int class_id,
  double yaw_world,
  double pitch_rad,
  const cv::Vec3d& tvec)
{
  const bool large =
    L2Perception::armorClassFromId(class_id)
      == L2Perception::ArmorClass::Hero;
  const double half_width = (large ? 0.230 : 0.135) / 2.0;
  const std::array<cv::Point3d, 4> object_points{{
    {0.0,  half_width,  0.0275},
    {0.0, -half_width,  0.0275},
    {0.0, -half_width, -0.0275},
    {0.0,  half_width, -0.0275}}};

  const Eigen::Matrix3d R_world_armor =
    Eigen::AngleAxisd(
      yaw_world,
      Eigen::Vector3d::UnitZ()).toRotationMatrix()
    * Eigen::AngleAxisd(
      pitch_rad,
      Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Matrix3d R_camera_armor =
    calibration.T_barrel_camera->linear().transpose()
    * R_world_armor;
  cv::Vec3d rvec;
  cv::Rodrigues(cvRotation(R_camera_armor), rvec);

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
  detection.class_id = class_id;
  detection.confidence = 1.0F;
  return detection;
}

void testModelTraitsAndConfig()
{
  const auto config =
    L3Estimation::loadL3Config("config/l3_config.yaml");
  require(
    L3Estimation::targetModelTraits(
      L3Estimation::TargetModel::FourArmorVehicle).armor_count == 4,
    "four-armor model traits are wrong");
  require(
    L3Estimation::targetModelTraits(
      L3Estimation::TargetModel::ThreeArmorOutpost).armor_count == 3,
    "outpost model traits are wrong");
  require(
    std::abs(
      config.armor.four_armor_vehicle.pitch_rad
      - 15.0 * std::numbers::pi / 180.0) < 1e-12,
    "vehicle pitch is not stored in radians");
  require(
    std::abs(
      config.armor.three_armor_outpost.initial_radius_m
      - 0.2765) < 1e-12
      && config.trackerConfig(
           L3Estimation::TargetModel::ThreeArmorOutpost)
           .angular_acceleration_variance == 0.1,
    "outpost-specific parameters were not loaded");
  require(
    config.tracker.enable_vehicle_geometry_constraints
      && std::abs(
      config.tracker.minimum_four_armor_corner_angle_rad
      - 50.0 * std::numbers::pi / 180.0) < 1e-12
      && std::abs(config.tracker.association_radius_gate - 0.20) < 1e-12
      && std::abs(config.tracker.association_radius_weight - 0.5) < 1e-12,
    "vehicle radius geometry parameters were not loaded");
}

void testVehicleEstimator()
{
  const auto calibration = makeCalibration();
  auto config =
    L3Estimation::loadL3Config("config/l3_config.yaml");
  config.tracker.confirmation_hits = 1;
  const L3Estimation::BarrelPoseProvider pose_provider =
    [](L3Estimation::TimePoint)
      -> std::optional<Eigen::Quaterniond> {
    return Eigen::Quaterniond::Identity();
  };
  L3Estimation::TargetEstimator estimator{
    calibration,
    pose_provider,
    config};
  const auto timestamp =
    L3Estimation::TimePoint{} + std::chrono::seconds{10};
  const int class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry3);
  const auto detection = makeDetection(
    calibration,
    class_id,
    0.32,
    config.armor.four_armor_vehicle.pitch_rad,
    {0.05, -0.02, 3.0});

  const auto targets = estimator.update(
    {detection},
    L3Estimation::FrameContext{
      .timestamp = timestamp,
      .image_size = calibration.image_size});
  require(
    targets.size() == 1
      && targets.front().model
           == L3Estimation::TargetModel::FourArmorVehicle
      && targets.front().armor_count == 4
      && targets.front().tracker_state
           == L3Estimation::TrackerState::Tracking,
    "vehicle was not published by the baseline estimator");
  require(
    estimator.lastObservations().size() == 1,
    "single detection did not produce one observation");
  const auto& observation = estimator.lastObservations().front();
  require(
    observation.source_detection_index == 0
      && observation.position_world.allFinite()
      && observation.rpy_raw_world.allFinite()
      && observation.rpy_constrained_world.allFinite()
      && std::isfinite(observation.yaw_world)
      && observation.optimized_reprojection_error_px
           <= observation.raw_yaw_reprojection_error_px,
    "baseline PnP/yaw observation is invalid");
  require(
    (observation.position_world - Eigen::Vector3d{3.0, -0.05, 0.02})
        .norm()
      < 5.0e-3,
    "optimized observation position is inaccurate");
  require(
    std::abs(observation.rpy_constrained_world.x()) < 1e-12
      && std::abs(
           observation.rpy_constrained_world.y()
           - config.armor.four_armor_vehicle.pitch_rad) < 1e-12
      && std::abs(std::remainder(
           observation.rpy_constrained_world.z()
             - observation.yaw_world,
           2.0 * std::numbers::pi)) < 1e-12,
    "constrained world RPY does not match the L3 armor model");
  require(
    estimator.lastAssociationDiagnostics().size() == 1
      && estimator.lastAssociationDiagnostics().front().accepted
      && estimator.lastAssociationDiagnostics().front()
           .associated_face_id == 0,
    "baseline initialization diagnostic is invalid");

  auto ignored_detection = detection;
  ignored_detection.class_id =
    static_cast<int>(L2Perception::ArmorClass::BaseSmall);
  (void)estimator.update(
    {ignored_detection, detection},
    L3Estimation::FrameContext{
      .timestamp = timestamp + std::chrono::milliseconds{10},
      .image_size = calibration.image_size});
  require(
    estimator.lastObservations().size() == 1
      && estimator.lastObservations().front().source_detection_index == 1,
    "observation did not preserve its source detection index");

  bool rejected_mismatched_size = false;
  try {
    (void)estimator.update(
      {detection},
      L3Estimation::FrameContext{
        .timestamp = timestamp,
        .image_size = {640, 512}});
  } catch (const std::invalid_argument&) {
    rejected_mismatched_size = true;
  }
  require(
    rejected_mismatched_size,
    "frame/calibration size mismatch was accepted");
}

void testOutpostEstimator()
{
  const auto calibration = makeCalibration();
  auto config =
    L3Estimation::loadL3Config("config/l3_config.yaml");
  config.tracker.confirmation_hits = 1;
  const L3Estimation::BarrelPoseProvider pose_provider =
    [](L3Estimation::TimePoint)
      -> std::optional<Eigen::Quaterniond> {
    return Eigen::Quaterniond::Identity();
  };
  L3Estimation::TargetEstimator estimator{
    calibration,
    pose_provider,
    config};
  const auto timestamp =
    L3Estimation::TimePoint{} + std::chrono::seconds{10};
  const auto detection = makeDetection(
    calibration,
    static_cast<int>(L2Perception::ArmorClass::Outpost),
    -0.25,
    config.armor.three_armor_outpost.pitch_rad,
    {-0.04, 0.01, 3.2});
  const auto targets = estimator.update(
    {detection},
    L3Estimation::FrameContext{
      .timestamp = timestamp,
      .image_size = calibration.image_size});
  require(
    targets.size() == 1
      && targets.front().model
           == L3Estimation::TargetModel::ThreeArmorOutpost
      && targets.front().armor_count == 3
      && std::abs(targets.front().radius - 0.2765) < 1e-12
      && targets.front().radius_offset == 0.0
      && targets.front().height_offset == 0.0,
    "outpost-specific baseline model is invalid");
}

void testDualCandidateSelection()
{
  const auto calibration = makeCalibration();
  auto config =
    L3Estimation::loadL3Config("config/l3_config.yaml");
  config.tracker.confirmation_hits = 1;
  const L3Estimation::BarrelPoseProvider pose_provider =
    [](L3Estimation::TimePoint)
      -> std::optional<Eigen::Quaterniond> {
    return Eigen::Quaterniond::Identity();
  };
  L3Estimation::TargetEstimator estimator{
    calibration,
    pose_provider,
    config};
  const auto timestamp =
    L3Estimation::TimePoint{} + std::chrono::seconds{10};
  const int class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry3);
  const double expected_yaw = 0.32;
  const auto detection = makeDetection(
    calibration,
    class_id,
    expected_yaw,
    config.armor.four_armor_vehicle.pitch_rad,
    {0.05, -0.02, 3.0});

  // 该近正视观测下两个 IPPE 候选都通过几何检查，存在镜像歧义。
  L3Estimation::PnpSolver solver{
    calibration,
    config.armor.dimensions,
    config.pnp};
  const auto poses =
    solver.solve(detection, L3Estimation::ArmorSize::Small);
  require(
    poses.size() == 2,
    "near-frontal detection did not produce two IPPE candidates");

  // 双候选按 yaw 优化误差选解后仍保持每检测一个观测，并恢复正确 yaw。
  const auto targets = estimator.update(
    {detection},
    L3Estimation::FrameContext{
      .timestamp = timestamp,
      .image_size = calibration.image_size});
  require(
    targets.size() == 1
      && targets.front().model
           == L3Estimation::TargetModel::FourArmorVehicle,
    "dual-candidate update did not publish one target");
  require(
    estimator.lastObservations().size() == 1,
    "dual-candidate update did not produce one observation");
  const auto& observation = estimator.lastObservations().front();
  require(
    std::abs(std::remainder(
      observation.yaw_world - expected_yaw,
      2.0 * std::numbers::pi))
      < 2.0 * std::numbers::pi / 180.0,
    "dual-candidate yaw selection did not recover the true yaw");
  require(
    observation.optimized_reprojection_error_px
      <= observation.raw_yaw_reprojection_error_px,
    "dual-candidate observation did not use the yaw-optimized pose");
}

void testPredictedFaceYawSelection()
{
  const auto calibration = makeCalibration();
  auto config =
    L3Estimation::loadL3Config("config/l3_config.yaml");
  config.tracker.confirmation_hits = 1;
  const L3Estimation::BarrelPoseProvider pose_provider =
    [](L3Estimation::TimePoint)
      -> std::optional<Eigen::Quaterniond> {
    return Eigen::Quaterniond::Identity();
  };
  L3Estimation::TargetEstimator estimator{
    calibration,
    pose_provider,
    config};
  const auto base_timestamp =
    L3Estimation::TimePoint{} + std::chrono::seconds{10};
  const int class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry3);
  const double expected_yaw = 0.32;

  // 第 1 帧初始化（无预测，走误差选解）并进入 Tracking。
  const auto first_detection = makeDetection(
    calibration,
    class_id,
    expected_yaw,
    config.armor.four_armor_vehicle.pitch_rad,
    {0.05, -0.02, 3.0});
  (void)estimator.update(
    {first_detection},
    L3Estimation::FrameContext{
      .timestamp = base_timestamp,
      .image_size = calibration.image_size});
  require(
    estimator.lastObservations().size() == 1,
    "first-frame observation was not produced");

  // 第 2 帧：已确认目标，预测 face yaw 选解仍输出单观测且 yaw 正确。
  const auto second_detection = makeDetection(
    calibration,
    class_id,
    expected_yaw,
    config.armor.four_armor_vehicle.pitch_rad,
    {0.05, -0.02, 3.0});
  (void)estimator.update(
    {second_detection},
    L3Estimation::FrameContext{
      .timestamp = base_timestamp + std::chrono::milliseconds{10},
      .image_size = calibration.image_size});
  require(
    estimator.lastObservations().size() == 1,
    "predicted-face selection did not keep one observation");
  const auto& predicted_observation =
    estimator.lastObservations().front();
  require(
    std::abs(std::remainder(
      predicted_observation.yaw_world - expected_yaw,
      2.0 * std::numbers::pi))
      < 2.0 * std::numbers::pi / 180.0,
    "predicted-face selection did not recover the true yaw");

  // 新 robot_id 无 Tracker：回退到误差选解，仍正常出观测。
  const int new_class_id =
    static_cast<int>(L2Perception::ArmorClass::Hero);
  const auto new_detection = makeDetection(
    calibration,
    new_class_id,
    -0.1,
    config.armor.four_armor_vehicle.pitch_rad,
    {0.0, 0.0, 3.5});
  (void)estimator.update(
    {new_detection},
    L3Estimation::FrameContext{
      .timestamp = base_timestamp + std::chrono::milliseconds{20},
      .image_size = calibration.image_size});
  require(
    estimator.lastObservations().size() == 1
      && estimator.lastObservations().front().robot_id == new_class_id,
    "new target fallback did not produce an observation");
}

void testPredictedFaceYawSelectionDisabled()
{
  const auto calibration = makeCalibration();
  auto config =
    L3Estimation::loadL3Config("config/l3_config.yaml");
  config.tracker.confirmation_hits = 1;
  config.pnp.enable_predicted_face_yaw_selection = false;
  const L3Estimation::BarrelPoseProvider pose_provider =
    [](L3Estimation::TimePoint)
      -> std::optional<Eigen::Quaterniond> {
    return Eigen::Quaterniond::Identity();
  };
  L3Estimation::TargetEstimator estimator{
    calibration,
    pose_provider,
    config};
  const auto base_timestamp =
    L3Estimation::TimePoint{} + std::chrono::seconds{10};
  const int class_id =
    static_cast<int>(L2Perception::ArmorClass::Infantry3);
  const auto detection = makeDetection(
    calibration,
    class_id,
    0.32,
    config.armor.four_armor_vehicle.pitch_rad,
    {0.05, -0.02, 3.0});
  (void)estimator.update(
    {detection},
    L3Estimation::FrameContext{
      .timestamp = base_timestamp,
      .image_size = calibration.image_size});
  (void)estimator.update(
    {detection},
    L3Estimation::FrameContext{
      .timestamp = base_timestamp + std::chrono::milliseconds{10},
      .image_size = calibration.image_size});
  require(
    estimator.lastObservations().size() == 1,
    "disabled predicted-face selection still produced one observation");
}

}  // namespace

int main()
{
  try {
    testModelTraitsAndConfig();
    testVehicleEstimator();
    testOutpostEstimator();
    testDualCandidateSelection();
    testPredictedFaceYawSelection();
    testPredictedFaceYawSelectionDisabled();
  } catch (const std::exception& error) {
    std::cerr << "L3 baseline smoke test failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "L3 baseline smoke test passed\n";
  return 0;
}
