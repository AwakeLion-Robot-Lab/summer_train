// L5 火控判定的冒烟测试。
//
// 这一层的设计取自工作空间里所有认真做火控的项目的共识：把装甲板的物理尺寸
// 换算成该距离上的角度容差，拿**实际**云台角去比，yaw 和 pitch 都比。用例
// 逐条盯住这些取舍，避免以后被"简化"回单一固定角度。

#include "l5_control/fire_decision.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <string>

namespace {

void require(bool condition, const std::string& message)
{
  if (!condition) {
    std::cerr << "fire decision smoke test failed: " << message << '\n';
    std::exit(1);
  }
}

[[nodiscard]] bool hasReason(
  const L5Control::FireDecision& decision, L5Control::RejectReason reason)
{
  for (const auto& item : decision.reasons) {
    if (item == reason) return true;
  }
  return false;
}

// 所有闸门都打开、云台完全对准的一帧。各用例在此基础上只破坏一个条件。
[[nodiscard]] L5Control::FireConfig makeConfig()
{
  L5Control::FireConfig config;
  config.shoot_enable = true;
  config.bullet_diameter = 0.017;
  config.min_bullet_speed = 21.0;
  config.max_bullet_speed = 30.0;
  config.heat_limit = 200.0;
  config.min_yaw = -std::numbers::pi;
  config.max_yaw = std::numbers::pi;
  config.min_pitch = -0.6;
  config.max_pitch = 0.6;
  return config;
}

[[nodiscard]] L5Control::FireInput makeInput()
{
  L5Control::FireInput input;

  // 火控只读目标的 name（用来查板型换算角度容差），滤波器状态本身用不到，
  // 所以这里用默认构造的目标就够。跟踪状态由 Tracker 单独提供。
  L3Estimation::TrackedTarget target;
  target.name = L3Estimation::ArmorName::Infantry3;
  input.target = target;
  input.track_state = L3Estimation::TrackState::Tracking;

  L4Planning::Plan plan;
  plan.valid = true;
  plan.ballistic_valid = true;
  plan.fire_admissible = true;
  plan.error = L4Planning::PlanError::None;
  plan.yaw = 0.0;
  plan.pitch = 0.05;
  plan.fire_armor_id = 0;
  plan.fire_delta_angle = 0.0;
  plan.fire_armor_point = {4.0, 0.0, 0.1};
  input.plan = plan;

  input.robot_state.heat = 0.0;
  input.actual_yaw = 0.0;
  input.actual_pitch = 0.05;
  input.calibration_ready = true;
  input.serial_fresh = true;
  input.gimbal_pose_fresh = true;
  return input;
}

void testAlignedShotIsAdmitted()
{
  const L5Control::FireDecider decider(makeConfig());
  const auto decision = decider.decide(makeInput());
  require(decision.reasons.empty(), "a fully aligned frame must have no reject reason");
  require(decision.fire_feasible, "a fully aligned frame must be feasible");
  require(decision.shoot, "shoot_enable = true must let the shot through");
  std::cout << "  [ok] aligned shot admitted, tolerance yaw="
            << decision.tolerance.yaw * 57.3 << " deg pitch="
            << decision.tolerance.pitch * 57.3 << " deg\n";
}

// shoot_enable 是人为闸门，不是"这一枪不该打"。两者必须分开记录，否则验收阶段
// 没法在不真的开火的情况下观察火控时序。
void testShootEnableGatesOnlyTheOutput()
{
  auto config = makeConfig();
  config.shoot_enable = false;
  const L5Control::FireDecider decider(config);

  const auto decision = decider.decide(makeInput());
  require(decision.fire_feasible, "shoot_enable must not affect fire_feasible");
  require(!decision.shoot, "shoot_enable = false must block the actual output");
  require(
    hasReason(decision, L5Control::RejectReason::ShootDisabled),
    "shoot_disabled must still be recorded");
  std::cout << "  [ok] shoot_enable gates the output, not the judgement\n";
}

// 容差来自装甲板在该距离上张开的角度，所以必须随距离收紧，并停在下限上。
void testToleranceShrinksWithDistance()
{
  const L5Control::FireDecider decider(makeConfig());

  auto near_input = makeInput();
  near_input.plan.fire_armor_point = {1.5, 0.0, 0.0};
  auto far_input = makeInput();
  far_input.plan.fire_armor_point = {8.0, 0.0, 0.0};

  const auto near = decider.decide(near_input).tolerance;
  const auto far = decider.decide(far_input).tolerance;
  require(near.valid && far.valid, "both tolerances must be computable");
  require(near.yaw > far.yaw, "yaw tolerance must shrink with distance");
  require(near.pitch > far.pitch, "pitch tolerance must shrink with distance");

  // 极远处物理张角趋近 0，没有下限就永远开不了火。
  auto very_far = makeInput();
  very_far.plan.fire_armor_point = {80.0, 0.0, 0.0};
  const auto floored = decider.decide(very_far).tolerance;
  require(
    std::abs(floored.yaw - decider.config().min_yaw_tolerance) < 1e-12 &&
      std::abs(floored.pitch - decider.config().min_pitch_tolerance) < 1e-12,
    "tolerance must settle on the configured floor at long range");
  std::cout << "  [ok] tolerance shrinks 1.5 m " << near.yaw * 57.3 << " deg -> 8 m "
            << far.yaw * 57.3 << " deg, floors at " << floored.yaw * 57.3 << " deg\n";
}

// 大装甲板更宽，容差必须更松。板型从车辆类别推出，与 L3 PnP 共用同一份映射。
void testBigArmorGetsWiderYawTolerance()
{
  const L5Control::FireDecider decider(makeConfig());

  auto small = makeInput();
  auto big = makeInput();
  big.target->name = L3Estimation::ArmorName::Hero;

  const auto small_tolerance = decider.decide(small).tolerance;
  const auto big_tolerance = decider.decide(big).tolerance;
  require(
    big_tolerance.yaw > small_tolerance.yaw,
    "a big armor plate must widen the yaw tolerance");
  require(
    std::abs(big_tolerance.pitch - small_tolerance.pitch) < 1e-12,
    "plate height is the same, so pitch tolerance must not change");
  std::cout << "  [ok] big plate widens yaw tolerance to " << big_tolerance.yaw * 57.3
            << " deg\n";
}

// 斜对枪口的板投影更窄，容差必须跟着收。缺了这一项，最容易脱靶的姿态反而拿到
// 和正对时一样宽的容差。
void testTiltedArmorNarrowsYawTolerance()
{
  const L5Control::FireDecider decider(makeConfig());

  auto facing = makeInput();
  auto tilted = makeInput();
  tilted.plan.fire_delta_angle = 60.0 * std::numbers::pi / 180.0;

  const auto a = decider.decide(facing).tolerance;
  const auto b = decider.decide(tilted).tolerance;
  require(b.yaw < a.yaw, "a tilted plate must narrow the yaw tolerance");
  std::cout << "  [ok] 60 deg tilt narrows yaw tolerance " << a.yaw * 57.3 << " -> "
            << b.yaw * 57.3 << " deg\n";
}

// pitch 也要判：只判 yaw 的话，俯仰没跟上照样开火。
void testPitchErrorAlsoBlocks()
{
  const L5Control::FireDecider decider(makeConfig());

  auto yaw_off = makeInput();
  yaw_off.actual_yaw = 0.2;
  auto pitch_off = makeInput();
  pitch_off.actual_pitch = 0.25;

  const auto by_yaw = decider.decide(yaw_off);
  const auto by_pitch = decider.decide(pitch_off);
  require(
    !by_yaw.fire_feasible && hasReason(by_yaw, L5Control::RejectReason::AimError),
    "yaw error must block the shot");
  require(
    !by_pitch.fire_feasible && hasReason(by_pitch, L5Control::RejectReason::AimError),
    "pitch error must block the shot too");
  std::cout << "  [ok] both axes gate the shot\n";
}

// 瞄准误差和"窗口里没有板"是两回事，必须分别归因。
void testWindowAndAimAreSeparateReasons()
{
  const L5Control::FireDecider decider(makeConfig());

  auto out_of_window = makeInput();
  out_of_window.plan.fire_admissible = false;
  const auto decision = decider.decide(out_of_window);
  require(
    hasReason(decision, L5Control::RejectReason::OutsideHitWindow),
    "an inadmissible plan must report OutsideHitWindow");
  require(
    !hasReason(decision, L5Control::RejectReason::AimError),
    "a well-aimed gimbal must not also be blamed for the window");
  std::cout << "  [ok] window and aim errors stay attributable\n";
}

// 所有拒绝原因一次列全，不短路。
void testReasonsAreNotShortCircuited()
{
  auto config = makeConfig();
  config.shoot_enable = false;
  const L5Control::FireDecider decider(config);

  auto input = makeInput();
  input.track_state = L3Estimation::TrackState::TempLost;
  input.serial_fresh = false;
  input.gimbal_pose_fresh = false;
  input.command_jump = true;
  input.armor_switching = true;
  input.calibration_ready = false;
  input.robot_state.heat = 500.0;
  input.plan.fire_admissible = false;
  input.actual_yaw = 1.0;

  const auto decision = decider.decide(input);
  for (const auto reason :
       {L5Control::RejectReason::ShootDisabled, L5Control::RejectReason::TempLost,
        L5Control::RejectReason::RobotStateStale,
        L5Control::RejectReason::GimbalPoseStale, L5Control::RejectReason::CommandJump,
        L5Control::RejectReason::ArmorSwitching,
        L5Control::RejectReason::MissingCalibration,
        L5Control::RejectReason::HeatLimit, L5Control::RejectReason::OutsideHitWindow,
        L5Control::RejectReason::AimError}) {
    require(hasReason(decision, reason), "reason " + toString(reason) + " must be listed");
  }
  require(!decision.fire_feasible && !decision.shoot, "a broken frame must not fire");
  std::cout << "  [ok] " << decision.reasons.size()
            << " reject reasons reported without short-circuiting\n";
}

// 中心档瞄的是旋转圆上的代理点，没有实体板可判时不许开火。
void testMissingFireArmorBlocks()
{
  const L5Control::FireDecider decider(makeConfig());

  auto input = makeInput();
  input.plan.fire_armor_id = -1;
  const auto decision = decider.decide(input);
  require(!decision.tolerance.valid, "no physical armor means no tolerance");
  require(
    !decision.fire_feasible && hasReason(decision, L5Control::RejectReason::AimError),
    "a plan without a physical fire armor must not fire");
  std::cout << "  [ok] center-proxy frames without a physical plate are refused\n";
}

// 未标定的火控参数必须拦住开火，不能拿默认值凑合。
void testMissingParametersBlock()
{
  L5Control::FireConfig config;
  config.shoot_enable = true;  // 参数没齐时，即使打开开关也不许开火
  const L5Control::FireDecider decider(config);

  const auto decision = decider.decide(makeInput());
  require(
    hasReason(decision, L5Control::RejectReason::MissingCalibration),
    "unfilled fire parameters must be reported");
  require(!decision.shoot, "unfilled fire parameters must block the shot");
  std::cout << "  [ok] missing fire parameters block the shot\n";
}

}  // namespace

int main()
{
  testAlignedShotIsAdmitted();
  testShootEnableGatesOnlyTheOutput();
  testToleranceShrinksWithDistance();
  testBigArmorGetsWiderYawTolerance();
  testTiltedArmorNarrowsYawTolerance();
  testPitchErrorAlsoBlocks();
  testWindowAndAimAreSeparateReasons();
  testReasonsAreNotShortCircuited();
  testMissingFireArmorBlocks();
  testMissingParametersBlock();

  std::cout << "fire decision smoke test passed\n";
  return 0;
}
