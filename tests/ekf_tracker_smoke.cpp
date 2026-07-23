#include "l3_estimation/ekf_tracker.hpp"

#include <Eigen/Eigenvalues>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using L3Estimation::ArmorObservation;
using L3Estimation::EkfTracker;
using L3Estimation::EkfTrackerConfig;
using L3Estimation::TargetState;
using L3Estimation::TimePoint;
using L3Estimation::TrackerState;

constexpr int kRobotId = 3;
constexpr double kRadius = 0.20;

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

double normalizeAngle(double angle)
{
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  angle = std::remainder(angle, kTwoPi);
  return angle <= -std::numbers::pi ? angle + kTwoPi : angle;
}

TimePoint timestampAt(int milliseconds)
{
  return TimePoint{} + std::chrono::seconds{1}
         + std::chrono::milliseconds{milliseconds};
}

ArmorObservation makeObservation(
  const Eigen::Vector3d& center,
  double yaw,
  int face_id,
  TimePoint timestamp,
  double radius = kRadius,
  double radius_offset = 0.0,
  double height_offset = 0.0,
  float confidence = 1.0F)
{
  const double face_yaw = normalizeAngle(
    yaw + static_cast<double>(face_id) * std::numbers::pi / 2.0);
  const bool second_group = face_id % 2 != 0;
  const double face_radius = radius + (second_group ? radius_offset : 0.0);

  return ArmorObservation{
    .robot_id = kRobotId,
    .armor_class = L2Perception::ArmorClass::Infantry3,
    .position_world = {
      center.x() - face_radius * std::cos(face_yaw),
      center.y() - face_radius * std::sin(face_yaw),
      center.z() + (second_group ? height_offset : 0.0)},
    .yaw_raw_world = face_yaw,
    .yaw_world = face_yaw,
    .confidence = confidence,
    .reprojection_error_px = 0.0,
    .timestamp = timestamp};
}

void requireValidCovariance(const TargetState& state)
{
  require(state.covariance.allFinite(), "covariance contains a non-finite value");
  require(
    (state.covariance - state.covariance.transpose()).norm() < 1e-8,
    "covariance is not symmetric");

  Eigen::SelfAdjointEigenSolver<L3Estimation::StateCovariance> eigen_solver(
    state.covariance,
    Eigen::EigenvaluesOnly);
  require(eigen_solver.info() == Eigen::Success, "covariance eigensolver failed");
  require(
    eigen_solver.eigenvalues().minCoeff() >= -1e-8,
    "covariance is not positive semidefinite");
}

void testInitializationAndConfirmation()
{
  const Eigen::Vector3d center{3.0, 0.5, 0.4};
  constexpr double yaw = 0.3;

  EkfTrackerConfig immediate_config;
  immediate_config.confirmation_hits = 1;
  EkfTracker immediate_tracker(kRobotId, immediate_config);
  immediate_tracker.update({makeObservation(center, yaw, 0, timestampAt(0))});

  require(
    immediate_tracker.trackerState() == TrackerState::Tracking,
    "confirmation_hits=1 did not publish the initialized tracker");
  require(
    (immediate_tracker.state().center - center).norm() < 1e-9,
    "vehicle center initialization uses the wrong normal direction");
  require(
    std::abs(immediate_tracker.state().radius - kRadius) < 1e-12,
    "initial radius mismatch");

  EkfTracker tracker(kRobotId);
  const ArmorObservation first = makeObservation(center, yaw, 0, timestampAt(0));
  tracker.update({first});
  for (int repeat = 0; repeat < 5; ++repeat) {
    tracker.update({first});
  }
  require(
    tracker.trackerState() == TrackerState::Detecting,
    "multiple updates at one timestamp counted as multiple frames");
  require(
    tracker.state().timestamp == TimePoint{},
    "unconfirmed tracker was published");

  for (int frame = 1; frame < 5; ++frame) {
    const TimePoint timestamp = timestampAt(frame * 20);
    tracker.predict(timestamp);
    tracker.update({makeObservation(center, yaw, 0, timestamp)});
  }
  require(
    tracker.trackerState() == TrackerState::Tracking,
    "tracker did not confirm after five consecutive frames");
  require(
    tracker.state().timestamp == timestampAt(80),
    "confirmed state has the wrong timestamp");
  requireValidCovariance(tracker.state());
}

void testStaticAndConstantVelocityTracking()
{
  const Eigen::Vector3d static_center{3.2, -0.4, 0.55};
  constexpr double static_yaw = -0.35;
  EkfTracker static_tracker(kRobotId);

  for (int frame = 0; frame < 60; ++frame) {
    const TimePoint timestamp = timestampAt(frame * 20);
    if (frame != 0) {
      static_tracker.predict(timestamp);
    }
    static_tracker.update({
      makeObservation(static_center, static_yaw, 0, timestamp)});
  }

  require(
    static_tracker.trackerState() == TrackerState::Tracking,
    "static target never entered tracking");
  require(
    (static_tracker.state().center - static_center).norm() < 0.03,
    "static center did not converge");
  require(
    static_tracker.state().velocity.norm() < 0.08,
    "static velocity did not converge near zero");
  require(
    std::abs(static_tracker.state().yaw_rate) < 0.08,
    "static yaw rate did not converge near zero");
  requireValidCovariance(static_tracker.state());

  EkfTrackerConfig moving_config;
  moving_config.confirmation_hits = 1;
  EkfTracker moving_tracker(kRobotId, moving_config);
  const Eigen::Vector3d initial_center{2.5, 0.3, 0.45};
  const Eigen::Vector3d velocity{0.45, -0.18, 0.08};
  constexpr double moving_yaw = 0.5;

  for (int frame = 0; frame < 80; ++frame) {
    const double elapsed = frame * 0.02;
    const TimePoint timestamp = timestampAt(frame * 20);
    const Eigen::Vector3d center = initial_center + velocity * elapsed;
    if (frame != 0) {
      moving_tracker.predict(timestamp);
    }
    moving_tracker.update({
      makeObservation(center, moving_yaw, 0, timestamp)});
  }

  const Eigen::Vector3d expected_center =
    initial_center + velocity * (79.0 * 0.02);
  require(
    (moving_tracker.state().center - expected_center).norm() < 0.06,
    "constant-velocity center tracking error is too large");
  require(
    (moving_tracker.state().velocity - velocity).norm() < 0.15,
    "constant velocity did not converge");
  requireValidCovariance(moving_tracker.state());
}

void testRotatingTargetAcrossPi()
{
  EkfTrackerConfig config;
  config.confirmation_hits = 1;
  EkfTracker tracker(kRobotId, config);

  const Eigen::Vector3d center{3.0, 0.2, 0.5};
  constexpr double initial_yaw = 3.0;
  constexpr double yaw_rate = 1.2;

  for (int frame = 0; frame < 100; ++frame) {
    const double elapsed = frame * 0.02;
    const TimePoint timestamp = timestampAt(frame * 20);
    const double yaw = normalizeAngle(initial_yaw + yaw_rate * elapsed);
    if (frame != 0) {
      tracker.predict(timestamp);
    }
    tracker.update({makeObservation(center, yaw, 0, timestamp)});
  }

  const double expected_yaw = normalizeAngle(initial_yaw + yaw_rate * 1.98);
  require(
    std::abs(normalizeAngle(tracker.state().yaw - expected_yaw)) < 0.08,
    "yaw became discontinuous while crossing pi");
  require(
    std::abs(tracker.state().yaw_rate - yaw_rate) < 0.25,
    "yaw rate did not converge across pi");
  requireValidCovariance(tracker.state());
}

void testDualArmorGeometryAndOutlierGate()
{
  EkfTrackerConfig config;
  config.confirmation_hits = 1;
  EkfTracker tracker(kRobotId, config);

  const Eigen::Vector3d center{3.5, -0.2, 0.45};
  constexpr double yaw = 0.25;
  constexpr double radius_offset = 0.07;
  constexpr double height_offset = 0.08;

  for (int frame = 0; frame < 80; ++frame) {
    const TimePoint timestamp = timestampAt(frame * 20);
    if (frame != 0) {
      tracker.predict(timestamp);
    }
    tracker.update({
      makeObservation(
        center, yaw, 0, timestamp,
        kRadius, radius_offset, height_offset, 1.0F),
      makeObservation(
        center, yaw, 1, timestamp,
        kRadius, radius_offset, height_offset, 0.9F)});
  }

  require(
    std::abs(tracker.state().radius_offset - radius_offset) < 0.035,
    "odd-face radius offset did not converge");
  require(
    std::abs(tracker.state().height_offset - height_offset) < 0.025,
    "odd-face height offset did not converge");
  requireValidCovariance(tracker.state());

  const TimePoint outlier_time = timestampAt(1600);
  tracker.predict(outlier_time);
  const TargetState predicted = tracker.state();
  ArmorObservation outlier = makeObservation(center, yaw, 0, outlier_time);
  outlier.position_world = {50.0, 50.0, 50.0};
  outlier.yaw_world = normalizeAngle(yaw + std::numbers::pi);
  tracker.update({outlier});

  require(
    tracker.trackerState() == TrackerState::TemporaryLost,
    "outlier incorrectly restored the tracker to Tracking");
  require(
    (tracker.state().center - predicted.center).norm() < 1e-12,
    "outlier changed the predicted state");
}

void testTemporaryLossExpirationAndBadTimestamps()
{
  EkfTrackerConfig config;
  config.confirmation_hits = 1;
  const Eigen::Vector3d center{2.8, 0.1, 0.5};
  constexpr double yaw = -0.2;

  EkfTracker tracker(kRobotId, config);
  tracker.update({makeObservation(center, yaw, 0, timestampAt(0))});
  tracker.predict(timestampAt(20));
  require(
    tracker.trackerState() == TrackerState::TemporaryLost,
    "one missed frame did not enter TemporaryLost");
  require(
    tracker.state().timestamp == timestampAt(20),
    "temporary-lost state was not predicted to the frame timestamp");
  require(!tracker.expired(timestampAt(300)), "tracker expired too early");
  require(tracker.expired(timestampAt(301)), "tracker did not expire after 300 ms");

  EkfTracker backward_tracker(kRobotId, config);
  backward_tracker.update({makeObservation(center, yaw, 0, timestampAt(20))});
  backward_tracker.predict(timestampAt(19));
  require(
    backward_tracker.trackerState() == TrackerState::Lost,
    "backward timestamp did not reset the tracker");
  require(
    backward_tracker.state().timestamp == TimePoint{},
    "reset tracker still exposes a timestamp");

  EkfTracker large_dt_tracker(kRobotId, config);
  large_dt_tracker.update({makeObservation(center, yaw, 0, timestampAt(0))});
  large_dt_tracker.predict(timestampAt(101));
  require(
    large_dt_tracker.trackerState() == TrackerState::Lost,
    "predict interval over 100 ms did not reset the tracker");
}

}  // namespace

int main()
{
  try {
    testInitializationAndConfirmation();
    testStaticAndConstantVelocityTracking();
    testRotatingTargetAcrossPi();
    testDualArmorGeometryAndOutlierGate();
    testTemporaryLossExpirationAndBadTimestamps();
  } catch (const std::exception& error) {
    std::cerr << "EkfTracker smoke test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "EkfTracker smoke test passed\n";
  return 0;
}
