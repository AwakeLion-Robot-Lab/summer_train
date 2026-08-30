// L5 火控判定的冒烟测试。
//
// 把装甲板的物理尺寸换算成该距离上的角度容差，再与实际云台 yaw/pitch 比较。
// 用例逐条固定这些关键判据，避免退化成单一固定角度。

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

bool hasReason(
  const L5Control::FireDecision& decision, L5Control::RejectReason reason)
{
  for (const auto& item : decision.reasons) {
    if (item == reason) return true;
  }
  return false;
}

// 所有闸门都打开、云台完全对准的一帧。各用例在此基础上只破坏一个条件。
L5Control::FireConfig makeConfig()
{
  L5Control::FireConfig config;
  config.shoot_enable = true;
  return config;
}

L5Control::FireInput makeInput()
{
  L5Control::FireInput input;

  // 火控只读目标的 name（用来查板型换算角度容差），滤波器状态本身用不到，
  // 所以用确定性构造入口给一个最简目标即可。跟踪状态由 Tracker 单独提供。
  L3Estimation::TrackedTarget target(
    L3Estimation::ArmorName::Infantry3, 4.0, 0.0, 0.2);
  input.target = target;
  input.track_state = L3Estimation::TrackState::Tracking;

  L4Planning::Plan plan;
  plan.status = L4Planning::PlanStatus::FireReady;
  plan.reason = L4Planning::PlanError::None;
  plan.aim = {{4.0, 0.0, 0.1}, 0.0, 0.05};
  plan.fire = L4Planning::FireReference{0, {4.0, 0.0, 0.1, 0.0}};
  input.plan = plan;

  input.actual_yaw = 0.0;
  input.actual_pitch = 0.05;
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
  near_input.plan.fire->armor_pose.head<3>() = Eigen::Vector3d{1.5, 0.0, 0.0};
  auto far_input = makeInput();
  far_input.plan.fire->armor_pose.head<3>() = Eigen::Vector3d{8.0, 0.0, 0.0};

  const auto near = decider.decide(near_input).tolerance;
  const auto far = decider.decide(far_input).tolerance;
  require(near.valid && far.valid, "both tolerances must be computable");
  require(near.yaw > far.yaw, "yaw tolerance must shrink with distance");
  require(near.pitch > far.pitch, "pitch tolerance must shrink with distance");

  // 极远处物理张角趋近 0，没有下限就永远开不了火。
  auto very_far = makeInput();
  very_far.plan.fire->armor_pose.head<3>() = Eigen::Vector3d{80.0, 0.0, 0.0};
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
  tilted.plan.fire->armor_pose.w() = 60.0 * std::numbers::pi / 180.0;

  const auto a = decider.decide(facing).tolerance;
  const auto b = decider.decide(tilted).tolerance;
  require(b.yaw < a.yaw, "a tilted plate must narrow the yaw tolerance");
  std::cout << "  [ok] 60 deg tilt narrows yaw tolerance " << a.yaw * 57.3 << " -> "
            << b.yaw * 57.3 << " deg\n";
}

// 板面误差必须是 wrap(line_of_sight - armor_normal)，不能只保存 armor yaw，
// 也不能拿车体中心方向代替这块板自己的视线方向。
void testFacingAngleUsesLineOfSight()
{
  constexpr double degree = std::numbers::pi / 180.0;

  L4Planning::FireReference ordinary{
    0, {4.0 * std::cos(40.0 * degree), 4.0 * std::sin(40.0 * degree), 0.0,
        25.0 * degree}};
  require(
    std::abs(ordinary.facingAngle() - 15.0 * degree) < 1e-12,
    "facing angle must subtract the armor normal from its own line of sight");

  L4Planning::FireReference wrapped{
    0, {4.0 * std::cos(-179.0 * degree), 4.0 * std::sin(-179.0 * degree), 0.0,
        179.0 * degree}};
  require(
    std::abs(wrapped.facingAngle() - 2.0 * degree) < 1e-12,
    "facing angle must wrap across the +/-pi boundary");
  std::cout << "  [ok] facing angle uses wrapped LOS - armor normal\n";
}

// pitch 与 yaw 都必须进入容差，任意一轴未到位都不能开火。
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

// 云台机械范围不在视觉火控层重复限制。
void testGimbalRangeIsNotFireGate()
{
  const L5Control::FireDecider decider(makeConfig());
  auto input = makeInput();
  input.plan.aim.yaw = 4.0;
  input.plan.aim.pitch = 1.0;
  input.actual_yaw = input.plan.aim.yaw;
  input.actual_pitch = input.plan.aim.pitch;

  const auto decision = decider.decide(input);
  require(
    decision.fire_feasible && decision.shoot,
    "gimbal software ranges must not gate firing in L5");
  std::cout << "  [ok] gimbal ranges are not L5 fire gates\n";
}

// 瞄准误差和"窗口里没有板"是两回事，必须分别归因。
void testWindowAndAimAreSeparateReasons()
{
  const L5Control::FireDecider decider(makeConfig());

  auto out_of_window = makeInput();
  out_of_window.plan.status = L4Planning::PlanStatus::TrackOnly;
  out_of_window.plan.reason = L4Planning::PlanError::OutOfWindow;
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
  input.command_jump = true;
  input.plan.status = L4Planning::PlanStatus::TrackOnly;
  input.plan.reason = L4Planning::PlanError::OutOfWindow;
  input.actual_yaw = 1.0;

  const auto decision = decider.decide(input);
  for (const auto reason :
       {L5Control::RejectReason::ShootDisabled, L5Control::RejectReason::TempLost,
        L5Control::RejectReason::CommandJump,
        L5Control::RejectReason::OutsideHitWindow, L5Control::RejectReason::AimError}) {
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
  input.plan.fire.reset();
  const auto decision = decider.decide(input);
  require(!decision.tolerance.valid, "no physical armor means no tolerance");
  require(
    !decision.fire_feasible && hasReason(decision, L5Control::RejectReason::AimError),
    "a plan without a physical fire armor must not fire");
  std::cout << "  [ok] center-proxy frames without a physical plate are refused\n";
}

// 降级原因必须精确：弹速异常不是击发窗口异常；整个 Plan 无效也不等于弹道失败。
void testPlanReasonsStayPrecise()
{
  const L5Control::FireDecider decider(makeConfig());

  auto bad_speed = makeInput();
  bad_speed.plan.status = L4Planning::PlanStatus::TrackOnly;
  bad_speed.plan.reason = L4Planning::PlanError::BadBulletSpeed;
  const auto speed_decision = decider.decide(bad_speed);
  require(
    hasReason(speed_decision, L5Control::RejectReason::BadBulletSpeed) &&
      !hasReason(speed_decision, L5Control::RejectReason::OutsideHitWindow),
    "bad bullet speed must not masquerade as an armor-window failure");

  // 延迟链没标完：只跟随不开火，而且要报成自己的原因，不能混进 PlanInvalid。
  auto uncalibrated = makeInput();
  uncalibrated.plan.status = L4Planning::PlanStatus::TrackOnly;
  uncalibrated.plan.reason = L4Planning::PlanError::DelayNotCalibrated;
  const auto uncalibrated_decision = decider.decide(uncalibrated);
  require(
    hasReason(uncalibrated_decision, L5Control::RejectReason::DelayNotCalibrated) &&
      !hasReason(uncalibrated_decision, L5Control::RejectReason::PlanInvalid) &&
      !uncalibrated_decision.fire_feasible,
    "an uncalibrated delay chain must block firing under its own reason");

  auto no_target = makeInput();
  no_target.plan.status = L4Planning::PlanStatus::Rejected;
  no_target.plan.reason = L4Planning::PlanError::NoTarget;
  const auto rejected_decision = decider.decide(no_target);
  require(
    hasReason(rejected_decision, L5Control::RejectReason::PlanInvalid) &&
      !hasReason(rejected_decision, L5Control::RejectReason::BallisticInvalid) &&
      !hasReason(rejected_decision, L5Control::RejectReason::OutsideHitWindow),
    "a rejected plan must retain its actual cause");
  std::cout << "  [ok] plan reject reasons stay precise\n";
}

}  // namespace

// 竖直命中窗口要随"装甲板后仰角 α + 视线仰角 β"收缩，可见高度是 h·|cos(α+β)|。
void testVerticalWindowFollowsPlateTilt()
{
  // 用近距离目标并放开 min_pitch_tolerance：默认 0.5 度的下限在 4 m 处会把
  // 倾角带来的差别整个夹平，那样测不出公式有没有接上。
  L5Control::FireConfig config;
  config.min_pitch_tolerance = 1e-6;
  const L5Control::FireDecider decider{config};

  // 同一个点、同一板型，只有类别不同——前哨站的板反着倾，于是只有 α 的符号变了。
  const auto windowAt = [&decider](double height_m, L3Estimation::ArmorName name) {
    L4Planning::Plan plan;
    plan.status = L4Planning::PlanStatus::FireReady;
    plan.fire = L4Planning::FireReference{0, {1.5, 0.0, height_m, 0.0}};
    return decider.tolerance(plan, L3Estimation::ArmorType::Small, name).pitch;
  };

  const auto kInfantry = L3Estimation::ArmorName::Infantry3;
  const auto kOutpost = L3Estimation::ArmorName::Outpost;

  // 抬头看：常规板后仰 +15 度与视线仰角叠加，窗口收窄；前哨站板前倾，两者
  // 部分抵消，同一位置反而看得更全。几何完全相同，差别只来自 α 的符号。
  require(
    windowAt(0.75, kOutpost) > windowAt(0.75, kInfantry),
    "抬头看时前倾的前哨板应当比后仰的常规板留出更大的竖直窗口");

  // 低头看：符号反过来。
  require(
    windowAt(-0.75, kInfantry) > windowAt(-0.75, kOutpost),
    "低头看时后仰的常规板反而更正对枪口");

  // 俯角恰好抵消后仰（α + β = 0）时看到完整板高，是这块板的窗口上界。
  const double aligned = windowAt(-1.5 * std::tan(15.0 * std::numbers::pi / 180.0), kInfantry);
  require(aligned > windowAt(0.75, kInfantry), "α+β=0 应当给出更大的窗口");
  require(aligned > windowAt(1.5, kInfantry), "偏离越多窗口越窄");

  // 视线接近与板面平行时窗口趋于 0，必须由 min_pitch_tolerance 兜住。
  const L5Control::FireConfig defaults;
  const L5Control::FireDecider clamped{defaults};
  L4Planning::Plan grazing;
  grazing.status = L4Planning::PlanStatus::FireReady;
  grazing.fire = L4Planning::FireReference{0, {1.5, 0.0, 8.0, 0.0}};
  require(
    clamped.tolerance(grazing, L3Estimation::ArmorType::Small, kInfantry).pitch >=
      defaults.min_pitch_tolerance,
    "掠射时必须退到最小 pitch 容差而不是 0");
  std::cout << "  [ok] vertical window follows plate tilt and line of sight\n";
}

int main()
{
  testVerticalWindowFollowsPlateTilt();
  testAlignedShotIsAdmitted();
  testShootEnableGatesOnlyTheOutput();
  testToleranceShrinksWithDistance();
  testBigArmorGetsWiderYawTolerance();
  testTiltedArmorNarrowsYawTolerance();
  testFacingAngleUsesLineOfSight();
  testPitchErrorAlsoBlocks();
  testGimbalRangeIsNotFireGate();
  testWindowAndAimAreSeparateReasons();
  testReasonsAreNotShortCircuited();
  testMissingFireArmorBlocks();
  testPlanReasonsStayPrecise();

  std::cout << "fire decision smoke test passed\n";
  return 0;
}
