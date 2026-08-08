// L4 规划链路的行为冒烟测试：整车外推、弹道模型与反解、瞄准档位迟滞、
// 逐板不动点迭代、选板与火控分离。不依赖相机、串口和推理后端。

#include "l4_planning/aim_phase.hpp"
#include "l4_planning/ballistic_model.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/planner.hpp"
#include "l4_planning/predictor.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

namespace {

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "planner smoke test failed: " << message << '\n';
    std::exit(1);
  }
}

L3Estimation::TargetState makeTarget(double v_yaw)
{
  L3Estimation::TargetState target;
  target.name = L3Estimation::ArmorName::Infantry3;
  target.track_state = L3Estimation::TrackState::Tracking;
  target.position = {4.0, 0.0, 0.0};
  target.velocity = Eigen::Vector3d::Zero();
  target.yaw = 0.0;
  target.v_yaw = v_yaw;
  target.radius = 0.2;
  target.armor_num = 4;
  target.second_radius = 0.2;
  target.height_diff = 0.0;
  // 默认已经观测到过第二块板，否则所有测试都会被可观测性门禁挡在 0 号板。
  target.multi_armor_observed = true;
  target.timestamp = std::chrono::steady_clock::time_point{};
  return target;
}

// 只推中心不推 yaw 是改造前的缺陷：小陀螺目标会被算成原地不动。
void testPredictorAdvancesYaw()
{
  const L4Planning::Predictor predictor;
  const auto target = makeTarget(10.0);

  const auto later = predictor.predict(target, 0.1);
  require(std::abs(later.yaw - 1.0) < 1e-9, "yaw must advance by v_yaw * dt");

  const auto before = predictor.armorPoses(target);
  const auto after = predictor.armorPosesAt(target, 0.1);
  require(before.size() == 4 && after.size() == 4, "four armor plates expected");

  const double moved = (before[0].head<3>() - after[0].head<3>()).norm();
  require(moved > 0.1, "spinning target armor must move over 100 ms");
  std::cout << "  [ok] predictor advances yaw, armor moved " << moved << " m\n";
}

// 平动目标：中心按速度走，装甲板整体跟着平移。
void testPredictorTranslates()
{
  const L4Planning::Predictor predictor;
  auto target = makeTarget(0.0);
  target.velocity = {1.0, 0.0, 0.0};

  const auto later = predictor.predict(target, 0.5);
  require(std::abs(later.position.x() - 4.5) < 1e-9, "center must translate");
  require(std::abs(later.yaw) < 1e-12, "yaw must stay put with zero v_yaw");
  std::cout << "  [ok] predictor translates a non-spinning target\n";
}

// 真空模型：解出的 pitch 代回抛体方程必须还原目标高度。
void testBallisticRoundTrip()
{
  const L4Planning::BallisticSolver solver;
  constexpr double kGravity = 9.7833;
  const double v0 = 23.0;

  for (const double distance : {1.5, 4.0, 7.0}) {
    for (const double height : {-0.5, 0.0, 0.8}) {
      const auto result = solver.solve(distance, height, v0);
      require(result.valid, "vacuum ballistic must be solvable at short range");

      // z = d*tan(theta) - g*d² / (2*v0²*cos²theta)
      const double cos_pitch = std::cos(result.pitch);
      const double reconstructed =
        distance * std::tan(result.pitch) -
        kGravity * distance * distance / (2.0 * v0 * v0 * cos_pitch * cos_pitch);
      require(
        std::abs(reconstructed - height) < 1e-6,
        "solved pitch must reproduce the target height");

      const double expected_time = distance / (v0 * cos_pitch);
      require(
        std::abs(result.fly_time - expected_time) < 1e-9,
        "fly time must match the horizontal component");
    }
  }
  std::cout << "  [ok] vacuum ballistic round-trips\n";
}

// 阻力系数为 0 时阻力模型必须严格退化成真空模型，否则默认配置一改就会引入
// 静默的弹道偏差。
void testDragDegradesToVacuum()
{
  const L4Planning::VacuumModel vacuum(9.7833);
  const L4Planning::QuadraticDragModel drag(9.7833, 0.0);

  for (const double pitch : {-0.2, 0.0, 0.05, 0.3}) {
    const auto a = vacuum.impact(5.0, pitch, 23.0);
    const auto b = drag.impact(5.0, pitch, 23.0);
    require(a.has_value() && b.has_value(), "both models must solve");
    require(std::abs(a->z - b->z) < 1e-12, "zero drag must match vacuum height");
    require(
      std::abs(a->fly_time - b->fly_time) < 1e-12, "zero drag must match vacuum time");

    const auto la = vacuum.launch(5.0, 0.2, 23.0);
    const auto lb = drag.launch(5.0, 0.2, 23.0);
    require(la.has_value() && lb.has_value(), "both inverses must solve");
    require(std::abs(la->pitch - lb->pitch) < 1e-12, "zero drag must match vacuum pitch");
  }
  std::cout << "  [ok] quadratic drag degrades to vacuum at k=0\n";
}

// 有阻力时子弹更慢、掉得更多，所以必须抬得更高、飞得更久。
void testDragNeedsMorePitch()
{
  L4Planning::BallisticConfig config;
  const L4Planning::BallisticSolver vacuum_solver(config);
  config.drag_coefficient = 0.02;
  const L4Planning::BallisticSolver drag_solver(config);

  const auto vacuum = vacuum_solver.solve(6.0, 0.0, 23.0);
  const auto drag = drag_solver.solve(6.0, 0.0, 23.0);
  require(vacuum.valid && drag.valid, "both solvers must converge at 6 m");
  require(drag.pitch > vacuum.pitch, "drag must require a higher muzzle angle");
  require(drag.fly_time > vacuum.fly_time, "drag must lengthen the flight");
  std::cout << "  [ok] quadratic drag raises pitch by "
            << (drag.pitch - vacuum.pitch) * 57.3 << " deg at 6 m\n";
}

// 反解是精确闭式解，不是迭代出来的近似。代回正向模型的残差必须落在机器精度，
// 而不是 height_tolerance —— 后者是没有闭式解时才允许的兜底精度。
//
// 这条用例盯住的是一个具体的退化：FYT / talos 对同一个模型跑高度补偿迭代，在
// 容差内提前退出，k=0.092、6 m 处留下约 0.02° 的系统性偏差；而且迭代次数随 k
// 和距离增长，10 m 处要 12 次，再远就会撞上 max_iterations 直接报无解。
void testDragInverseIsExact()
{
  for (const double k : {0.0, 0.019, 0.092}) {
    L4Planning::BallisticConfig config;
    config.drag_coefficient = k;
    const L4Planning::BallisticSolver solver(config);

    for (const double distance : {1.5, 4.0, 6.0, 10.0}) {
      for (const double height : {-0.4, 0.0, 0.6}) {
        const auto result = solver.solve(distance, height, 23.0);
        require(result.valid, "closed form must solve inside the envelope");

        const auto impact = solver.model().impact(distance, result.pitch, 23.0);
        require(impact.has_value(), "forward model must accept the solved pitch");
        require(
          std::abs(impact->z - height) < 1e-9,
          "closed-form inverse must be exact, not merely within tolerance");
        require(
          std::abs(impact->fly_time - result.fly_time) < 1e-12,
          "fly time must agree between forward and inverse");
      }
    }
  }
  std::cout << "  [ok] drag inverse is exact to 1e-9 m at k = 0 / 0.019 / 0.092\n";
}

// 没有闭式反解的模型必须自动走高度补偿迭代，且解出来和闭式解一致。这条用例
// 就是"将来加 RK4 全阻力模型不用改求解器"这句话的凭据。
class ForwardOnlyDragModel final : public L4Planning::IBallisticModel {
public:
  ForwardOnlyDragModel(double gravity, double drag) noexcept : inner_(gravity, drag) {}

  [[nodiscard]] std::optional<L4Planning::Impact> impact(
    double range, double pitch, double v0) const noexcept override
  {
    return inner_.impact(range, pitch, v0);
  }
  // 刻意不覆盖 launch()，落到基类的 nullopt 上。
  [[nodiscard]] std::string_view name() const noexcept override
  {
    return "forward_only";
  }

private:
  L4Planning::QuadraticDragModel inner_;
};

void testIterativeFallbackMatchesClosedForm()
{
  const ForwardOnlyDragModel forward_only(9.7833, 0.02);
  require(
    !forward_only.launch(6.0, 0.2, 23.0).has_value(),
    "a forward-only model must not advertise a closed form");

  L4Planning::BallisticConfig config;
  config.drag_coefficient = 0.02;
  const L4Planning::BallisticSolver closed_form(config);

  // 手工跑一遍求解器的兜底路径，确认它收敛到同一个角度。
  const L4Planning::VacuumModel seed(config.gravity);
  double aim_height = 0.2;
  double pitch = 0.0;
  for (int i = 0; i < config.max_iterations; ++i) {
    const auto guess = seed.launch(6.0, aim_height, 23.0);
    require(guess.has_value(), "seed must solve");
    pitch = guess->pitch;
    const auto impact = forward_only.impact(6.0, pitch, 23.0);
    require(impact.has_value(), "forward model must solve");
    const double error = 0.2 - impact->z;
    if (std::abs(error) < config.height_tolerance) break;
    aim_height += error;
  }

  const auto exact = closed_form.solve(6.0, 0.2, 23.0);
  require(exact.valid, "closed form must solve");
  require(
    std::abs(pitch - exact.pitch) < 2e-3,
    "iterative fallback must land near the closed-form solution");
  std::cout << "  [ok] iterative fallback tracks the closed form within "
            << std::abs(pitch - exact.pitch) * 57.3 << " deg\n";
}

void testBallisticRejectsBadInput()
{
  const L4Planning::BallisticSolver solver;
  require(!solver.solve(4.0, 0.0, 0.0).valid, "zero bullet speed must be rejected");
  require(!solver.solve(0.0, 0.0, 23.0).valid, "zero distance must be rejected");
  require(!solver.solve(500.0, 0.0, 23.0).valid, "out-of-range target must be rejected");
  require(!solver.solve(std::nan(""), 0.0, 23.0).valid, "NaN distance must be rejected");

  // 业务门限已经全部收归 PlanConfig，求解器不得再自带一道弹速门槛，否则
  // 落在两道门限夹缝里的弹速会被报成 BallisticFailed 而不是 BadBulletSpeed。
  require(
    solver.solve(4.0, 0.0, 15.0).valid,
    "solver must not impose a business-level bullet speed threshold");
  std::cout << "  [ok] ballistic rejects unusable input, keeps no business gate\n";
}

// 档位切换必须又钝又带回差，否则 v_yaw 在阈值附近抖一次就换一次策略。
void testAimPhaseHysteresis()
{
  L4Planning::AimPhaseConfig config;
  config.transfer_count = 5;
  L4Planning::AimPhaseTracker tracker(config);

  require(
    tracker.phase() == L4Planning::AimPhase::SingleArmor, "must start at SingleArmor");

  // 超过上行阈值但没坚持够帧数：不换档。
  for (int i = 0; i < config.transfer_count; ++i) tracker.update(3.0, true);
  require(
    tracker.phase() == L4Planning::AimPhase::SingleArmor,
    "transfer must need more than transfer_count frames");

  tracker.update(3.0, true);
  require(
    tracker.phase() == L4Planning::AimPhase::WholeCarArmor, "must reach WholeCarArmor");

  // 落回死区（single_to_whole_down 与 single_to_whole_up 之间）不该回退。
  for (int i = 0; i < 50; ++i) tracker.update(1.2, true);
  require(
    tracker.phase() == L4Planning::AimPhase::WholeCarArmor,
    "dead band must not trigger a downgrade");

  // 真正掉到下行阈值以下才回退。
  for (int i = 0; i <= config.transfer_count; ++i) tracker.update(0.5, true);
  require(
    tracker.phase() == L4Planning::AimPhase::SingleArmor,
    "below the down threshold must downgrade");

  // 高速升到中心档。
  for (int i = 0; i <= config.transfer_count; ++i) tracker.update(20.0, true);
  require(tracker.phase() == L4Planning::AimPhase::WholeCarArmor, "first rung");
  for (int i = 0; i <= config.transfer_count; ++i) tracker.update(20.0, true);
  require(
    tracker.phase() == L4Planning::AimPhase::WholeCarCenter, "must reach WholeCarCenter");

  // 几何不可观测时无条件退回最保守档位。
  tracker.update(20.0, false);
  require(
    tracker.phase() == L4Planning::AimPhase::SingleArmor,
    "unobservable geometry must force SingleArmor");
  std::cout << "  [ok] aim phase ladder holds its hysteresis\n";
}

// 不动点迭代应当收敛，且命中时刻的瞄准点确实由飞行时间决定。
void testPlannerConverges()
{
  L4Planning::Planner planner;
  const auto target = makeTarget(0.0);

  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  const auto plan = planner.plan(target, robot_state, target.timestamp);
  require(plan.valid, "static target must be plannable");
  require(plan.error == L4Planning::PlanError::None, "no rejection expected");
  require(plan.fire_admissible, "a static facing target must be shootable");
  require(plan.ballistic_valid, "ballistic must be valid");
  require(plan.fly_time > 0.0, "fly time must be positive");
  require(
    std::abs(plan.delay.fire_to_hit - plan.fly_time) < 1e-12,
    "fire_to_hit must carry the fly time");
  require(plan.hit_time > plan.fire_time, "hit must come after fire");
  require(plan.aim_on_armor, "low speed must aim at a physical plate");

  // 目标在 x 轴正方向，选中的板朝向枪口，yaw 应当接近 0。
  require(std::abs(plan.yaw) < 0.1, "yaw should point at the target");
  require(plan.pitch > 0.0, "muzzle must be raised to hit a level target");
  std::cout << "  [ok] planner converges, fly_time=" << plan.fly_time
            << " pitch=" << plan.pitch << '\n';
}

// 弹速无效时仍要给出瞄准角，但必须把 BadBulletSpeed 报给 L5。
void testPlannerFlagsBadBulletSpeed()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 0.0;

  const auto plan = planner.plan(makeTarget(0.0), robot_state, {});
  require(plan.valid, "fallback speed must still produce a plan");
  require(!plan.fire_admissible, "bad bullet speed must block firing");
  require(
    plan.error == L4Planning::PlanError::BadBulletSpeed,
    "bad bullet speed must be reported");

  // 15 m/s 落在旧的两道门限（PlanConfig 10、求解器 21）之间，正是会被报成
  // BallisticFailed 的那段。门限收归一处后必须报 BadBulletSpeed。
  robot_state.bullet_speed = 15.0;
  const auto between = planner.plan(makeTarget(0.0), robot_state, {});
  require(between.valid, "a low bullet speed must still yield aim angles");
  require(
    between.error == L4Planning::PlanError::BadBulletSpeed,
    "the 10-21 m/s band must report BadBulletSpeed, not BallisticFailed");

  // 上限同样要判。只判下限的话，裁判系统回传一个明显偏高的弹速会被直接采信，
  // 解出来的弹道偏平，而且没有任何人报错。
  robot_state.bullet_speed = 45.0;
  const auto too_fast = planner.plan(makeTarget(0.0), robot_state, {});
  require(too_fast.valid, "an absurdly high bullet speed must still yield aim angles");
  require(!too_fast.fire_admissible, "an absurdly high bullet speed must block firing");
  require(
    too_fast.error == L4Planning::PlanError::BadBulletSpeed,
    "bullet speed above the upper bound must report BadBulletSpeed");
  std::cout << "  [ok] planner flags bad bullet speed on both bounds but still aims\n";
}

void testPlannerRejectsNoTarget()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  const auto empty = planner.plan(std::nullopt, robot_state, {});
  require(!empty.valid, "missing target must not produce a plan");
  require(empty.error == L4Planning::PlanError::NoTarget, "NoTarget expected");

  auto lost = makeTarget(0.0);
  lost.track_state = L3Estimation::TrackState::Lost;
  const auto lost_plan = planner.plan(lost, robot_state, {});
  require(!lost_plan.valid, "lost target must not produce a plan");
  require(lost_plan.error == L4Planning::PlanError::NotTracking, "NotTracking expected");
  std::cout << "  [ok] planner rejects missing and lost targets\n";
}

// 只见过一块板时整车 yaw、第二组半径、高度差都还没被观测约束过，瞄别的板
// 等于拿伪造的几何开火。
void testUnobservedGeometryLocksArmorZero()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  auto target = makeTarget(0.0);
  target.multi_armor_observed = false;
  // 把整车转到 1 号板正对枪口的姿态：几何可观测时会选 1 号。
  target.yaw = -std::numbers::pi / 2.0;

  const auto blind = planner.plan(target, robot_state, {});
  require(blind.valid, "an unobserved target must still be trackable");
  require(blind.armor_id == 0, "unobserved geometry must stay on armor 0");

  planner.reset();
  target.multi_armor_observed = true;
  const auto seen = planner.plan(target, robot_state, {});
  require(seen.valid, "observed target must be plannable");
  require(seen.armor_id == 1, "observed geometry must be free to pick armor 1");
  std::cout << "  [ok] unobserved geometry pins the aim to armor 0\n";
}

// 选板迟滞：两块板对称时不应逐帧互换。
void testSelectorHysteresis()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  // yaw 让 0 号和 3 号板相对枪口近似对称，正是容易横跳的构型。
  auto target = makeTarget(0.0);
  target.yaw = std::numbers::pi / 4.0;

  const auto first = planner.plan(target, robot_state, {});
  require(first.valid, "symmetric configuration must still be plannable");
  const int locked = planner.lockedArmorId();
  require(locked >= 0, "an armor must be locked");

  // 轻微扰动 yaw，锁定的板不应改变。
  int switches = 0;
  for (int step = 0; step < 20; ++step) {
    target.yaw = std::numbers::pi / 4.0 + (step % 2 == 0 ? 1e-3 : -1e-3);
    const auto plan = planner.plan(target, robot_state, {});
    require(plan.valid, "perturbed plan must stay valid");
    if (plan.armor_id != locked) ++switches;
  }
  require(switches == 0, "hysteresis must suppress armor flapping");
  std::cout << "  [ok] selector hysteresis holds across 20 perturbed frames\n";
}

// 反陀螺现在是**火控**判据而不是选板判据：整圈扫下来云台必须帧帧有角度，
// 而可开火的只占其中一段。旧实现在窗口外直接不给瞄准角，云台会停摆。
void testFireWindowGatesWithoutDroppingAim()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  auto target = makeTarget(8.0);
  int valid_count = 0;
  int fire_count = 0;
  for (int step = 0; step < 60; ++step) {
    target.yaw = L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 60.0);
    const auto plan = planner.plan(target, robot_state, {});
    if (plan.valid) ++valid_count;
    if (plan.fire_admissible) {
      ++fire_count;
      require(
        plan.fire_armor_id >= 0 && plan.fire_armor_id < 4,
        "fire armor id must be in range");
      require(
        plan.error == L4Planning::PlanError::None,
        "an admissible frame must not carry a degradation reason");
    }
  }
  require(valid_count == 60, "the gimbal must keep tracking on every frame");
  require(fire_count > 0, "the fire window must open somewhere in a full turn");
  require(fire_count < 60, "the fire window must also close");
  std::cout << "  [ok] aim valid 60/60, fire admissible " << fire_count << "/60\n";
}

// 高速档瞄旋转圆上离枪口最近的点，火控仍按实体板判定。
void testCenterPhaseProjectsInsideTheCircle()
{
  L4Planning::PlanConfig config;
  config.aim_phase.transfer_count = 2;
  L4Planning::Planner planner(config);
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  auto target = makeTarget(20.0);  // 高于 whole_to_center_up
  L4Planning::Plan plan;
  for (int step = 0; step < 40; ++step) {
    target.yaw = L6Telemetry::limit_rad(step * 0.3);
    plan = planner.plan(target, robot_state, {});
  }

  require(
    planner.aimPhase() == L4Planning::AimPhase::WholeCarCenter,
    "sustained high spin must reach the center phase");
  require(plan.valid, "center phase must still produce angles");
  require(!plan.aim_on_armor, "center phase does not aim at a physical plate");

  // 代理点应当落在旋转中心与枪口之间，且离中心恰好一个半径。
  const auto predicted = L4Planning::Predictor().predict(
    target, plan.delay.beforeFire() + plan.fly_time);
  const double to_center =
    (plan.aim_point.head<2>() - predicted.position.head<2>()).norm();
  require(
    std::abs(to_center - target.radius) < 1e-6,
    "center proxy must sit one radius from the rotation center");
  require(
    plan.aim_point.head<2>().norm() < predicted.position.head<2>().norm(),
    "center proxy must sit on the muzzle side of the center");
  require(plan.fire_armor_id >= 0, "fire gate must still name a physical plate");
  std::cout << "  [ok] center phase proxies " << target.radius
            << " m in front of the rotation center\n";
}

// 逐板求解的意义：选板放进迭代里时，飞行时间会在两块板之间来回跳而不收敛。
// 这里扫过整圈的所有构型，要求每一帧都能收敛出解。
void testPerArmorFixedPointAlwaysConverges()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  for (const double v_yaw : {0.5, 2.0, 6.0, 12.0}) {
    auto target = makeTarget(v_yaw);
    for (int step = 0; step < 120; ++step) {
      target.yaw = L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 120.0);
      const auto plan = planner.plan(target, robot_state, {});
      require(plan.valid, "every configuration must yield a converged plan");
      require(std::isfinite(plan.yaw) && std::isfinite(plan.pitch), "angles must be finite");
    }
    planner.reset();
  }
  std::cout << "  [ok] per-armor fixed point converges on 480 configurations\n";
}

}  // namespace

int main()
{
  testPredictorAdvancesYaw();
  testPredictorTranslates();
  testBallisticRoundTrip();
  testDragDegradesToVacuum();
  testDragNeedsMorePitch();
  testDragInverseIsExact();
  testIterativeFallbackMatchesClosedForm();
  testBallisticRejectsBadInput();
  testAimPhaseHysteresis();
  testPlannerConverges();
  testPlannerFlagsBadBulletSpeed();
  testPlannerRejectsNoTarget();
  testUnobservedGeometryLocksArmorZero();
  testSelectorHysteresis();
  testFireWindowGatesWithoutDroppingAim();
  testCenterPhaseProjectsInsideTheCircle();
  testPerArmorFixedPointAlwaysConverges();
  std::cout << "planner smoke test passed\n";
  return 0;
}
