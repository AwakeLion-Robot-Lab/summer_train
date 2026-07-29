#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/controller.hpp"
#include "l5_control/fire_decision.hpp"

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

  const L4Planning::AimPlan switching = run_observed(kPi / 2.0);
  if (!switching.valid || switching.armor_id != 3
      || switching.fire_permitted || !switching.armor_switching
      || switching.tracking_phase
           != L4Planning::ArmorTrackingPhase::Stabilizing) {
    std::cerr << "invalid current armor did not start a safe switch\n";
    return 3;
  }
  const L5Control::Controller controller;
  if (L5Control::shouldFire(switching)
      || controller.makeCommand(switching).shoot) {
    std::cerr << "downstream control ignored switch fire inhibition\n";
    return 3;
  }

  const L4Planning::AimPlan relocked = run_observed(kPi / 2.0);
  if (!relocked.valid || relocked.armor_id != 3
      || !relocked.fire_permitted || relocked.armor_switching
      || relocked.tracking_phase
           != L4Planning::ArmorTrackingPhase::Tracking) {
    std::cerr << "new armor did not finish stable locking\n";
    return 4;
  }
  if (!L5Control::shouldFire(relocked)
      || !controller.makeCommand(relocked).shoot) {
    std::cerr << "downstream control rejected a stable locked armor\n";
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
      || missing_once.armor_id != 3 || missing_twice.armor_id != 3) {
    std::cerr << "short target loss was not predicted safely\n";
    return 5;
  }

  const L4Planning::AimPlan lost = run_missing();
  if (lost.valid || lost.tracking
      || planner.trackingState().phase
           != L4Planning::ArmorTrackingPhase::Unlocked) {
    std::cerr << "tracking was not released after the loss limit\n";
    return 6;
  }

  L4Planning::Planner rotating_planner;
  L4Planning::PlannerContext rotating_context;
  rotating_context.config.lock_stable_frames = 2;
  rotating_context.config.switch_dead_zone = 5.0;
  L1Sensor::RobotState rotating_robot = robot_state;
  rotating_robot.rpy.yaw = 0.0;
  rotating_robot.rpy.pitch = 0.0;
  auto rotating_time = L4Planning::TimePoint{2s};
  const auto run_rotating =
    [&](double yaw) {
      rotating_time += 10ms;
      rotating_robot.timestamp = rotating_time;
      rotating_context.planning_time = rotating_time;
      return rotating_planner.plan(
        std::optional<L3Estimation::TargetState>{
          makeTarget(rotating_time, yaw, 1.0)},
        rotating_robot,
        rotating_context);
    };

  (void)run_rotating(-0.35);
  const L4Planning::AimPlan rotating_locked = run_rotating(-0.35);
  if (!rotating_locked.valid || rotating_locked.armor_id != 0
      || rotating_locked.tracking_phase
           != L4Planning::ArmorTrackingPhase::Tracking) {
    std::cerr << "rotating target did not establish the initial lock\n";
    return 7;
  }

  const L4Planning::AimPlan pre_switch = run_rotating(-0.10);
  if (!pre_switch.valid || pre_switch.armor_id != 0
      || !pre_switch.fire_permitted || pre_switch.armor_switching
      || pre_switch.tracking_phase
           != L4Planning::ArmorTrackingPhase::PreSwitch
      || !rotating_planner.trackingState().next_armor_id.has_value()
      || *rotating_planner.trackingState().next_armor_id != 3) {
    std::cerr << "outgoing armor did not prepare its adjacent successor\n";
    return 8;
  }

  rotating_robot.rpy.yaw = 1.0;
  const L4Planning::AimPlan rotating_switch = run_rotating(0.40);
  if (!rotating_switch.valid || rotating_switch.armor_id != 3
      || rotating_switch.fire_permitted || !rotating_switch.armor_switching
      || rotating_switch.tracking_phase
           != L4Planning::ArmorTrackingPhase::Switching) {
    std::cerr << "preselected armor did not enter the switching phase\n";
    return 9;
  }

  rotating_robot.rpy.yaw = 0.0;
  const L4Planning::AimPlan rotating_stabilizing = run_rotating(0.40);
  const L4Planning::AimPlan rotating_relocked = run_rotating(0.40);
  if (!rotating_stabilizing.valid
      || rotating_stabilizing.tracking_phase
           != L4Planning::ArmorTrackingPhase::Stabilizing
      || rotating_stabilizing.fire_permitted
      || !rotating_relocked.valid || rotating_relocked.armor_id != 3
      || rotating_relocked.tracking_phase
           != L4Planning::ArmorTrackingPhase::Tracking
      || !rotating_relocked.fire_permitted) {
    std::cerr << "rotating armor did not stabilize and relock\n";
    return 10;
  }

  std::cout << "Planner armor tracking smoke test passed\n";
  return 0;
}
