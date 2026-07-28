#include "l3_estimation/ekf_tracker.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {

constexpr int kRobotId =
  static_cast<int>(L2Perception::ArmorClass::Infantry3);

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

L3Estimation::TimePoint timestampAt(int milliseconds)
{
  return L3Estimation::TimePoint{}
         + std::chrono::seconds{10}
         + std::chrono::milliseconds{milliseconds};
}

L3Estimation::ArmorObservation makeObservation(
  L3Estimation::TargetModel model,
  const Eigen::Vector3d& center,
  double vehicle_yaw,
  int face_id,
  double radius,
  L3Estimation::TimePoint timestamp)
{
  const double face_yaw =
    vehicle_yaw
    + static_cast<double>(face_id)
        * L3Estimation::targetModelTraits(model)
            .face_angle_interval_rad;
  return {
    .robot_id =
      model == L3Estimation::TargetModel::ThreeArmorOutpost
        ? static_cast<int>(L2Perception::ArmorClass::Outpost)
        : kRobotId,
    .armor_class =
      model == L3Estimation::TargetModel::ThreeArmorOutpost
        ? L2Perception::ArmorClass::Outpost
        : L2Perception::ArmorClass::Infantry3,
    .model = model,
    .position_world = {
      center.x() - radius * std::cos(face_yaw),
      center.y() - radius * std::sin(face_yaw),
      center.z()},
    .yaw_raw_world = face_yaw,
    .yaw_world = face_yaw,
    .confidence = 1.0F,
    .pnp_reprojection_error_px = 0.2,
    .raw_yaw_reprojection_error_px = 0.2,
    .optimized_reprojection_error_px = 0.2,
    .timestamp = timestamp};
}

void testFourArmorTracker()
{
  L3Estimation::EkfTrackerConfig config;
  config.confirmation_hits = 2;
  config.initial_radius = 0.20;
  config.association_position_gate = 0.30;
  config.association_yaw_gate = 0.80;
  L3Estimation::EkfTracker tracker{
    kRobotId,
    L3Estimation::TargetModel::FourArmorVehicle,
    config};

  const Eigen::Vector3d center{3.0, 0.2, 0.5};
  constexpr double kYaw = 0.25;
  const auto first = tracker.update({
    makeObservation(
      L3Estimation::TargetModel::FourArmorVehicle,
      center,
      kYaw,
      0,
      0.20,
      timestampAt(0))});
  require(
    tracker.trackerState() == L3Estimation::TrackerState::Detecting
      && first.size() == 1
      && first.front().accepted,
    "first observation did not initialize Detecting");

  tracker.predict(timestampAt(20));
  const auto second = tracker.update({
    makeObservation(
      L3Estimation::TargetModel::FourArmorVehicle,
      center,
      kYaw,
      1,
      0.20,
      timestampAt(20))});
  require(
    tracker.trackerState() == L3Estimation::TrackerState::Tracking
      && tracker.state().timestamp == timestampAt(20),
    "second consecutive observation did not confirm Tracking");
  require(
    !second.empty()
      && second.front().accepted
      && second.front().associated_face_id == 1
      && second.front().nis_valid,
    "simple physical-face matching failed");
  require(
    second.front().lifecycle_before
        == L3Estimation::TrackerState::Detecting
      && second.front().lifecycle_after
           == L3Estimation::TrackerState::Tracking,
    "observed frame lifecycle is incorrect");

  tracker.predict(timestampAt(40));
  const auto missed = tracker.update({});
  require(
    tracker.trackerState()
        == L3Estimation::TrackerState::TemporaryLost
      && !missed.empty()
      && missed.front().lifecycle_before
           == L3Estimation::TrackerState::Tracking
      && missed.front().lifecycle_after
           == L3Estimation::TrackerState::TemporaryLost,
    "missing frame did not enter TemporaryLost at frame end");

  // 旧时间戳观测必须被忽略，不能倒灌更新当前滤波状态。
  const auto stale = tracker.update({
    makeObservation(
      L3Estimation::TargetModel::FourArmorVehicle,
      center,
      kYaw,
      0,
      0.20,
      timestampAt(0))});
  require(
    stale.empty()
      && !tracker.state().updated_this_frame
      && tracker.state().timestamp == timestampAt(40),
    "stale observation changed the current filter state");
}

void testOutpostConstraints()
{
  L3Estimation::EkfTrackerConfig config;
  config.confirmation_hits = 1;
  config.initial_radius = 0.2765;
  config.linear_acceleration_variance = 10.0;
  config.angular_acceleration_variance = 0.1;
  const int robot_id =
    static_cast<int>(L2Perception::ArmorClass::Outpost);
  L3Estimation::EkfTracker tracker{
    robot_id,
    L3Estimation::TargetModel::ThreeArmorOutpost,
    config};
  const Eigen::Vector3d center{3.2, -0.1, 0.55};
  (void)tracker.update({
    makeObservation(
      L3Estimation::TargetModel::ThreeArmorOutpost,
      center,
      -0.30,
      0,
      0.2765,
      timestampAt(0))});

  const auto& state = tracker.state();
  require(
    state.tracker_state == L3Estimation::TrackerState::Tracking
      && state.armor_count == 3,
    "outpost did not use the three-face model");
  require(
    state.radius_offset == 0.0
      && state.height_offset == 0.0
      && state.covariance
           .row(L3Estimation::RADIUS_OFFSET).isZero(0.0)
      && state.covariance
           .row(L3Estimation::HEIGHT_OFFSET).isZero(0.0),
    "outpost offset states were not locked");
}

}  // namespace

int main()
{
  try {
    testFourArmorTracker();
    testOutpostConstraints();
  } catch (const std::exception& error) {
    std::cerr << "EkfTracker smoke test failed: "
              << error.what() << '\n';
    return 1;
  }

  std::cout << "EkfTracker smoke test passed\n";
  return 0;
}
