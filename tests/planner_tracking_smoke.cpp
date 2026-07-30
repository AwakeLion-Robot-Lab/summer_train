#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/planner.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>

namespace {

constexpr double kPi = 3.14159265358979323846;

L3Estimation::TargetState makeTarget(
  L4Planning::TimePoint timestamp,
  double yaw,
  double yaw_rate = 0.0)
{
  L3Estimation::TargetState target;
  target.robot_id = 3;
  target.center = {5.0, 0.0, 0.0};
  target.velocity.setZero();
  target.yaw = yaw;
  target.yaw_rate = yaw_rate;
  target.radius = 0.25;
  target.radius_offset = 0.0;
  target.height_offset = 0.0;
  target.covariance =
    L3Estimation::StateCovariance::Identity() * 1e-3;
  target.timestamp = timestamp;
  return target;
}

}  // namespace

int main()
{
  using namespace std::chrono_literals;

  L4Planning::Planner planner;
  L4Planning::PlannerContext context;
  context.config.lock_stable_frames = 2;
  // 本测试只验证选择状态机，放宽枪口到候选弹道角的稳定阈值。
  context.config.switch_dead_zone = 180.0;
  context.config.max_lost_frames = 2;

  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 30.0;

  auto timestamp = L4Planning::TimePoint{1s};
  const auto run_observed =
    [&](double yaw) {
      timestamp += 10ms;
      robot_state.timestamp = timestamp;
      context.planning_time = timestamp;
      return planner.plan(
        std::optional<L3Estimation::TargetState>{
          makeTarget(timestamp, yaw)},
        robot_state,
        context);
    };

  const L4Planning::AimPlan first = run_observed(0.0);
  if (!first.valid || first.armor_id != 0 || first.fire_permitted
      || first.tracking_phase != L4Planning::ArmorTrackingPhase::Stabilizing) {
    std::cerr << "initial armor should enter stabilizing phase\n";
    return 1;
  }

  const L4Planning::AimPlan locked = run_observed(0.0);
  if (!locked.valid || locked.armor_id != 0 || !locked.fire_permitted
      || locked.tracking_phase != L4Planning::ArmorTrackingPhase::Tracking) {
    std::cerr << "initial armor did not become locked\n";
    return 2;
  }

  robot_state.timestamp = timestamp;
  context.planning_time = timestamp;
  const L4Planning::AimPlan repeated_timestamp = planner.plan(
    std::optional<L3Estimation::TargetState>{
      makeTarget(timestamp, 0.0)},
    robot_state,
    context);
  if (!repeated_timestamp.valid || repeated_timestamp.armor_id != 0
      || repeated_timestamp.fire_permitted
      || repeated_timestamp.tracking_phase
           != L4Planning::ArmorTrackingPhase::Tracking) {
    std::cerr << "repeated observation timestamp was treated as fresh\n";
    return 9;
  }

  for (int confirmation_frame = 1; confirmation_frame < 3;
       ++confirmation_frame) {
    const L4Planning::AimPlan pending_switch =
      run_observed(kPi / 2.0);
    if (!pending_switch.valid || pending_switch.armor_id != 0
        || pending_switch.tracking_phase
             != L4Planning::ArmorTrackingPhase::Tracking) {
      std::cerr << "score advantage switched armor before three frames\n";
      return 10;
    }
  }

  const L4Planning::AimPlan retained = run_observed(kPi / 2.0);
  if (!retained.valid || retained.armor_id == 0
      || retained.fire_permitted
      || retained.tracking_phase
           != L4Planning::ArmorTrackingPhase::Stabilizing) {
    std::cerr << "higher-scored armor did not start a stable switch\n";
    return 3;
  }
  const L4Planning::AimPlan retained_again = run_observed(kPi / 2.0);
  if (!retained_again.valid
      || retained_again.armor_id != retained.armor_id
      || !retained_again.fire_permitted
      || retained_again.tracking_phase
           != L4Planning::ArmorTrackingPhase::Tracking) {
    std::cerr << "higher-scored armor did not finish relocking\n";
    return 4;
  }
  const auto run_missing =
    [&]() {
      timestamp += 10ms;
      robot_state.timestamp = timestamp;
      context.planning_time = timestamp;
      return planner.plan(std::nullopt, robot_state, context);
    };
  const L4Planning::AimPlan missing_once = run_missing();
  const L4Planning::AimPlan missing_twice = run_missing();
  if (!missing_once.valid || !missing_twice.valid
      || missing_once.fire_permitted || missing_twice.fire_permitted
      || missing_once.armor_id != retained.armor_id
      || missing_twice.armor_id != retained.armor_id) {
    std::cerr << "stale tracking did not retain aim while inhibiting fire\n";
    return 5;
  }

  const L4Planning::AimPlan lost = run_missing();
  if (lost.valid || lost.tracking
      || planner.trackingState().phase
           != L4Planning::ArmorTrackingPhase::Unlocked) {
    std::cerr << "tracking was not released after the loss limit\n";
    return 6;
  }

  // 将阈值设为评分的完整量程，验证窗口状态本身不会强制换板。
  L4Planning::Planner threshold_planner;
  L4Planning::PlannerContext threshold_context;
  threshold_context.config.lock_stable_frames = 2;
  threshold_context.config.switch_dead_zone = 180.0;
  threshold_context.config.score_switch_threshold = 1.0;
  L1Sensor::RobotState threshold_robot = robot_state;
  auto threshold_time = L4Planning::TimePoint{2s};
  const auto run_threshold =
    [&](double yaw) {
      threshold_time += 10ms;
      threshold_robot.timestamp = threshold_time;
      threshold_context.planning_time = threshold_time;
      return threshold_planner.plan(
        std::optional<L3Estimation::TargetState>{
          makeTarget(threshold_time, yaw)},
        threshold_robot,
        threshold_context);
    };

  (void)run_threshold(0.0);
  const L4Planning::AimPlan threshold_locked = run_threshold(0.0);
  if (!threshold_locked.valid || threshold_locked.armor_id != 0
      || threshold_locked.tracking_phase
           != L4Planning::ArmorTrackingPhase::Tracking) {
    std::cerr << "threshold test did not establish the initial lock\n";
    return 7;
  }

  const L4Planning::AimPlan threshold_retained =
    run_threshold(kPi / 2.0);
  if (!threshold_retained.valid || threshold_retained.armor_id != 0
      || threshold_retained.fire_permitted
      || threshold_retained.tracking_phase
           != L4Planning::ArmorTrackingPhase::Tracking) {
    std::cerr << "window state forced a switch or allowed firing\n";
    return 8;
  }

  std::cout << "Planner armor tracking smoke test passed\n";
  return 0;
}
