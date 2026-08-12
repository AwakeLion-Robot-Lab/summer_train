// L4 规划链路的行为冒烟测试：基础预测/弹道组件，以及 Planner 对 SP Aimer
// 的延迟、锁板和共同命中时刻迭代复刻。不依赖相机、串口和推理后端。

#include "l4_planning/aim_phase.hpp"
#include "l4_planning/ballistic_model.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/planner.hpp"
#include "l4_planning/predictor.hpp"
#include "l6_telemetry/math.hpp"

#include <Eigen/Dense>

#include <chrono>
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

// 旋转中心固定在 (4, 0, 0)，半径 0.2，四板车。jumped 默认置真，否则所有测试
// 都会被可观测性门禁挡在 0 号板。
L3Estimation::TrackedTarget makeTarget(double v_yaw, double yaw = 0.0)
{
  L3Estimation::TrackedTarget target(4.0, v_yaw, 0.2, 0.0, yaw);
  target.name = L3Estimation::ArmorName::Infantry3;
  target.jumped = true;
  return target;
}

// 只推中心不推 yaw 是改造前的缺陷：小陀螺目标会被算成原地不动。
void testPredictorAdvancesYaw()
{
  const L4Planning::Predictor predictor;
  const auto target = makeTarget(10.0);

  const auto later = predictor.predict(target, 0.1);
  require(
    std::abs(later.ekf_x()[6] - 1.0) < 1e-9, "yaw must advance by v_yaw * dt");

  const auto before = predictor.armorPoses(target);
  const auto after = predictor.armorPosesAt(target, 0.1);
  require(before.size() == 4 && after.size() == 4, "four armor plates expected");

  const double moved = (before[0].head<3>() - after[0].head<3>()).norm();
  require(moved > 0.1, "spinning target armor must move over 100 ms");
  std::cout << "  [ok] predictor advances yaw, armor moved " << moved << " m\n";
}

// 平动目标：中心按速度走，装甲板整体跟着平移。平动速度只能由 EKF 从观测里估
// 出来——确定性构造入口给不出 vx，所以这里先喂一段匀速直线运动的观测。
void testPredictorTranslates()
{
  const L4Planning::Predictor predictor;
  const std::chrono::steady_clock::time_point t0{};

  L3Estimation::Armor observation;
  observation.name = L3Estimation::ArmorName::Infantry3;
  observation.type = L3Estimation::ArmorType::Small;
  observation.xyz_in_world = {3.8, 0.0, 0.0};
  observation.ypr_in_world = {0.0, 0.0, 0.0};
  observation.ypd_in_world = L6Telemetry::xyz2ypd(observation.xyz_in_world);

  Eigen::VectorXd P0(11);
  P0 << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0, 0.4, 100.0, 1.0, 1.0, 1.0;
  L3Estimation::TrackedTarget target(observation, t0, 0.2, 4, P0);

  // 十帧，每帧 20 ms 沿 +x 前进 2 cm，即 1 m/s。
  for (int step = 1; step <= 10; ++step) {
    observation.xyz_in_world.x() = 3.8 + 0.02 * step;
    observation.ypd_in_world = L6Telemetry::xyz2ypd(observation.xyz_in_world);
    target.predict(t0 + std::chrono::milliseconds(20 * step));
    target.update(observation);
  }
  require(target.ekf_x()[1] > 0.1, "EKF must pick up a positive x velocity");

  const Eigen::VectorXd before = target.ekf_x();
  const auto later = predictor.predict(target, 0.5);
  const Eigen::VectorXd after = later.ekf_x();
  require(
    std::abs(after[0] - (before[0] + before[1] * 0.5)) < 1e-9,
    "center must translate by velocity * dt");
  require(
    std::abs(
      after[6] - L6Telemetry::limit_rad(before[6] + before[7] * 0.5)) < 1e-9,
    "yaw must advance by exactly v_yaw * dt");
  std::cout << "  [ok] predictor translates a target at "
            << before[1] << " m/s\n";
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

  const auto plan = planner.plan(target, robot_state, target.t());
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
  require(plan.pitch < 0.0, "SP command pitch must negate the ballistic pitch");
  std::cout << "  [ok] planner converges, fly_time=" << plan.fly_time
            << " pitch=" << plan.pitch << '\n';
}

// SP 只在弹速 <14 m/s 时回退到 23；14 及以上不设上限。
void testPlannerMatchesSpBulletFallback()
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

  robot_state.bullet_speed = 13.99;
  const auto below = planner.plan(makeTarget(0.0), robot_state, {});
  require(below.valid, "SP fallback must still yield aim angles");
  require(
    below.error == L4Planning::PlanError::BadBulletSpeed,
    "speed below 14 m/s must use the SP fallback");

  for (const double speed : {14.0, 15.0, 45.0}) {
    robot_state.bullet_speed = speed;
    const auto accepted = planner.plan(makeTarget(0.0), robot_state, {});
    require(accepted.valid, "SP must accept every speed at or above 14 m/s");
    require(
      accepted.error == L4Planning::PlanError::None,
      "SP must not impose an upper bullet-speed gate");
  }
  std::cout << "  [ok] planner uses SP's one-sided 14 m/s fallback\n";
}

void testPlannerRejectsNoTarget()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  // 目标丢失由 Tracker 表达成"不返回目标"，所以规划器这一侧只剩空值这一种
  // 情况——不再有携带 Lost 状态的目标快照。
  const auto empty = planner.plan(std::nullopt, robot_state, {});
  require(!empty.valid, "missing target must not produce a plan");
  require(empty.error == L4Planning::PlanError::NoTarget, "NoTarget expected");

  // 默认构造的目标滤波器是空的，必须被 NoArmor 拦住而不是解引用空状态。
  const L3Estimation::TrackedTarget uninitialized;
  const auto uninitialized_plan = planner.plan(uninitialized, robot_state, {});
  require(!uninitialized_plan.valid, "an empty filter must not produce a plan");
  require(
    uninitialized_plan.error == L4Planning::PlanError::NoArmor,
    "NoArmor expected for an uninitialized target");
  std::cout << "  [ok] planner rejects missing and uninitialized targets\n";
}

// 只见过一块板时整车 yaw、第二组半径、高度差都还没被观测约束过，瞄别的板
// 等于拿伪造的几何开火。
void testUnobservedGeometryLocksArmorZero()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  // 把整车转到 1 号板正对枪口的姿态：几何可观测时会选 1 号。
  auto target = makeTarget(0.0, -std::numbers::pi / 2.0);
  target.jumped = false;

  const auto blind = planner.plan(target, robot_state, {});
  require(blind.valid, "an unobserved target must still be trackable");
  require(blind.armor_id == 0, "unobserved geometry must stay on armor 0");

  planner.reset();
  target.jumped = true;
  const auto seen = planner.plan(target, robot_state, {});
  require(seen.valid, "observed target must be plannable");
  require(seen.armor_id == 1, "observed geometry must be free to pick armor 1");
  std::cout << "  [ok] unobserved geometry pins the aim to armor 0\n";
}

// 复刻 SP 的锁板边界：两块板同时留在 ±60° 窗口内时，即使另一块
// 已经更正对枪口也不能提前换；只有旧板离开窗口才切到新板。
void testSpSelectorHoldsUntilArmorLeavesWindow()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  const auto degrees = [](double value) {
    return value * std::numbers::pi / 180.0;
  };

  const auto first = planner.plan(
    makeTarget(0.0, degrees(44.0)), robot_state, {});
  require(first.valid && first.armor_id == 0, "SP did not initially lock armor 0");

  // SP Aimer 没有 reset：无目标和显式 reset 都不能改变 lock_id。
  const auto no_target = planner.plan(std::nullopt, robot_state, {});
  require(!no_target.valid, "missing target must remain invalid");
  planner.reset();

  for (const double yaw_degrees : {50.0, 59.0}) {
    const auto plan = planner.plan(
      makeTarget(0.0, degrees(yaw_degrees)), robot_state, {});
    require(plan.valid, "overlap-window plan must stay valid");
    require(plan.armor_id == 0, "SP lock changed across no-target/reset or overlap");
  }

  const auto after_leaving = planner.plan(
    makeTarget(0.0, degrees(61.0)), robot_state, {});
  require(after_leaving.valid, "plan after leaving the window must stay valid");
  require(after_leaving.armor_id == 3, "SP lock did not switch after armor 0 left");

  const auto overlap_again = planner.plan(
    makeTarget(0.0, degrees(59.0)), robot_state, {});
  require(overlap_again.valid, "returning overlap-window plan must stay valid");
  require(overlap_again.armor_id == 3, "new SP lock was not retained in overlap");
  std::cout << "  [ok] SP selector holds a plate until it leaves the 60 deg window\n";
}

// SP 没有 AimPhase/WholeCarCenter；即使角速度很高，普通四板车仍瞄实体板。
void testSpAimerAlwaysAimsAtPhysicalArmor()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  for (int step = 0; step < 60; ++step) {
    const auto target = makeTarget(
      20.0, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 60.0));
    const auto plan = planner.plan(target, robot_state, {});
    require(plan.valid, "SP normal-car branch must keep producing an aim point");
    require(plan.aim_on_armor, "SP must never replace an armor with a center proxy");
    require(
      plan.aim_phase == L4Planning::AimPhase::SingleArmor,
      "inactive AimPhase metadata must remain SingleArmor");
    require(
      plan.armor_id >= 0 && plan.armor_id < 4,
      "SP selected armor id must be physical");
  }
  std::cout << "  [ok] SP high-speed path always aims at a physical armor\n";
}

void testSpDelaySelection()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  const auto low = planner.plan(makeTarget(8.0), robot_state, {}, false);
  const auto high = planner.plan(makeTarget(8.01), robot_state, {}, false);
  const auto negative = planner.plan(makeTarget(-20.0), robot_state, {}, false);
  require(low.valid && high.valid && negative.valid, "delay test plans must be valid");
  require(
    std::abs(low.delay.beforeFire() - 0.020) < 1e-12,
    "w == decision_speed must use SP low delay");
  require(
    std::abs(high.delay.beforeFire() - 0.035) < 1e-12,
    "positive w above decision_speed must use SP high delay");
  require(
    std::abs(negative.delay.beforeFire() - 0.020) < 1e-12,
    "negative high speed must still use SP low delay");

  const auto target = makeTarget(0.0);
  const auto to_now = planner.plan(
    target, robot_state, target.t() + std::chrono::milliseconds(12), true);
  require(to_now.valid, "to_now delay test plan must be valid");
  require(
    std::abs(to_now.delay.beforeFire() - 0.027) < 1e-12,
    "to_now must add measured elapsed time to SP low delay");
  std::cout << "  [ok] SP signed speed delay and offline 5 ms path match\n";
}

// SP 在每轮共同命中时刻重新选板；扫描普通四板车确认这条路径保持有限输出。
void testSpSharedIterationProducesFiniteCommands()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  for (const double v_yaw : {0.5, 2.0, 6.0, 12.0}) {
    for (int step = 0; step < 120; ++step) {
      const auto target = makeTarget(
        v_yaw, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 120.0));
      const auto plan = planner.plan(target, robot_state, {});
      require(plan.valid, "SP shared iteration must yield a command here");
      require(std::isfinite(plan.yaw) && std::isfinite(plan.pitch), "angles must be finite");
    }
    planner.reset();
  }
  std::cout << "  [ok] SP shared predict/choose iteration is finite on 480 configurations\n";
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
  testPlannerMatchesSpBulletFallback();
  testPlannerRejectsNoTarget();
  testUnobservedGeometryLocksArmorZero();
  testSpSelectorHoldsUntilArmorLeavesWindow();
  testSpAimerAlwaysAimsAtPhysicalArmor();
  testSpDelaySelection();
  testSpSharedIterationProducesFiniteCommands();
  std::cout << "planner smoke test passed\n";
  return 0;
}
