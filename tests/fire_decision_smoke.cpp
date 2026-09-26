#include "l5_control/fire_decision.hpp"
#include "l5_control/controller.hpp"
#include "l6_telemetry/math.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const std::string& message)
{
  if (!condition) {
    std::cerr << "fire decision smoke test failed: " << message << '\n';
    std::exit(1);
  }
}

bool hasReason(
  const L5Control::FireDecision& decision,
  L5Control::RejectReason reason)
{
  for (const auto item : decision.reasons) {
    if (item == reason) {
      return true;
    }
  }
  return false;
}

L5Control::FireConfig makeConfig(bool shoot_enable = true)
{
  L5Control::FireConfig config;
  config.shoot_enable = shoot_enable;
  return config;
}

L5Control::FireInput makeInput()
{
  L5Control::FireInput input;
  input.target.emplace(
    L3Estimation::ArmorName::Infantry3, 4.0, 0.0, 0.2);
  input.track_state = L3Estimation::TrackState::Tracking;
  input.plan.valid = true;
  input.plan.tracked = true;
  input.plan.fire_permitted = true;
  input.plan.armor_id = 0;
  input.plan.impact_time = input.target->t();
  input.plan.aim_point_barrel = {3.8, 0.0, 0.0};
  input.plan.aim_point_world = input.plan.aim_point_barrel;
  input.plan.yaw = 0.0;
  input.plan.pitch = 0.0;
  input.actual_yaw = 0.0;
  input.actual_pitch = 0.0;
  return input;
}

void testAlignedShot()
{
  const L5Control::FireDecider decider(makeConfig());
  const auto decision = decider.decide(makeInput());
  require(decision.reasons.empty(), "aligned plan must have no reject reason");
  require(decision.fire_feasible && decision.shoot, "aligned plan must fire");
}

void testPlannerWindowGate()
{
  const L5Control::FireDecider decider(makeConfig());
  auto input = makeInput();
  input.plan.fire_permitted = false;
  const auto decision = decider.decide(input);
  require(
    hasReason(decision, L5Control::RejectReason::OutsideHitWindow),
    "AimPlan::fire_permitted must gate firing");
  require(!decision.fire_feasible && !decision.shoot, "closed window must not fire");
}

void testInvalidPlan()
{
  const L5Control::FireDecider decider(makeConfig());
  auto input = makeInput();
  input.plan.valid = false;
  const auto decision = decider.decide(input);
  require(
    hasReason(decision, L5Control::RejectReason::PlanInvalid),
    "AimPlan::valid must gate firing");
}

void testNoUsableBulletSpeedNeverFires()
{
  const L5Control::FireDecider decider(makeConfig());
  auto input = makeInput();
  input.bullet_speed_valid = false;
  const auto decision = decider.decide(input);
  require(
    hasReason(decision, L5Control::RejectReason::BadBulletSpeed),
    "an unusable effective bullet speed must be reported");
  require(
    !decision.fire_feasible && !decision.shoot,
    "an unusable effective bullet speed must never permit firing");
}

void testNoUsableBulletSpeedStillTracks()
{
  L5Control::Controller controller(makeConfig(), 1.0);
  auto input = makeInput();
  const auto command = controller.update(
    input.target,
    input.track_state,
    input.plan,
    Eigen::Quaterniond::Identity(),
    false);
  require(
    command.has_value(),
    "an unusable effective bullet speed must still produce an aim command");
  require(
    command->yaw == input.plan.yaw && command->pitch == input.plan.pitch,
    "an unusable effective bullet speed must preserve planned aim angles");
  require(
    !command->shoot,
    "an unusable effective bullet speed must force shoot=false");
  require(
    hasReason(
      controller.lastDecision(), L5Control::RejectReason::BadBulletSpeed),
    "controller must retain the unusable bullet speed rejection");
}

void testControllerConvertsPitchFeedbackConvention()
{
  L5Control::Controller controller(makeConfig(), 1.0);
  auto input = makeInput();
  input.plan.pitch = 0.08;
  // 电控姿态的 Ry 俯仰角向下为正；物理上向上 0.08 rad
  // 因此以 -0.08 rad 姿态回传。L5 转换后应与规划角对齐。
  const Eigen::Quaterniond actual_pose = L6Telemetry::rpyToQuaternion(
    0.0, -input.plan.pitch, 0.0);
  const auto command = controller.update(
    input.target,
    input.track_state,
    input.plan,
    actual_pose,
    true);
  require(command.has_value(), "converted pitch feedback must produce a command");
  require(
    controller.lastDecision().pitch_error < 1e-9,
    "controller must compare pitch in the planning convention");
  require(
    controller.lastDecision().fire_feasible && command->shoot,
    "aligned nonzero pitch must permit firing after feedback conversion");
}

void testShootEnableOnlyGatesOutput()
{
  const L5Control::FireDecider decider(makeConfig(false));
  const auto decision = decider.decide(makeInput());
  require(
    hasReason(decision, L5Control::RejectReason::ShootDisabled),
    "disabled output must be reported");
  require(decision.fire_feasible && !decision.shoot,
          "shoot_enable must not hide theoretical feasibility");
}

void testMpcCommandAngles()
{
  const L5Control::FireDecider decider(makeConfig());
  auto input = makeInput();
  input.plan.using_MPC = true;
  input.plan.samples.push_back({});
  input.plan.samples.front().yaw = 0.02;
  input.plan.samples.front().pitch = -0.01;
  input.actual_yaw = 0.02;
  input.actual_pitch = -0.01;
  const auto decision = decider.decide(input);
  require(decision.fire_feasible, "MPC's first sample must drive the fire check");

  input.plan.samples.clear();
  const auto empty = decider.decide(input);
  require(
    hasReason(empty, L5Control::RejectReason::PlanInvalid),
    "an empty MPC trajectory must be rejected");
}

}  // namespace

int main()
{
  testAlignedShot();
  testPlannerWindowGate();
  testInvalidPlan();
  testNoUsableBulletSpeedNeverFires();
  testNoUsableBulletSpeedStillTracks();
  testControllerConvertsPitchFeedbackConvention();
  testShootEnableOnlyGatesOutput();
  testMpcCommandAngles();
  std::cout << "fire decision smoke test passed\n";
  return 0;
}
