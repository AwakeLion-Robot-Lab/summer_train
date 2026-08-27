// L4 规划链路的行为冒烟测试：覆盖预测、弹道、延迟、锁板和命中时刻迭代。
// 不依赖相机、串口和推理后端。

#include "l4_planning/ballistic.hpp"
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
// 防止闭式反解退化为提前退出的近似迭代：近似迭代会留下系统性角度残差，且可能
// 随距离增加撞上 max_iterations 后误报无解。
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

// 不动点迭代应当收敛，且命中时刻的瞄准点确实由飞行时间决定。
void testPlannerConverges()
{
  L4Planning::Planner planner;
  const auto target = makeTarget(0.0);

  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  const auto plan = planner.plan(target, robot_state, target.t());
  require(plan.valid(), "static target must be plannable");
  require(plan.reason == L4Planning::PlanError::None, "no rejection expected");
  require(plan.fireAdmissible(), "a static facing target must be shootable");
  require(plan.fire.has_value(), "valid physical aim must carry a fire reference");
  require(plan.timing.fly_time > 0.0, "fly time must be positive");
  require(
    std::abs(plan.timing.delay.fire_to_hit - plan.timing.fly_time) < 1e-12,
    "fire_to_hit must carry the fly time");
  require(
    plan.timing.prediction_time > target.t(),
    "prediction time must be after the source image");
  require(
    (plan.aim.point - plan.fire->point()).norm() < 1e-12,
    "setpoint planner must aim at its physical fire reference");

  // 目标在 x 轴正方向，选中的板朝向枪口，yaw 应当接近 0。
  require(std::abs(plan.aim.yaw) < 0.1, "yaw should point at the target");
  require(plan.aim.pitch < 0.0, "command pitch must follow the world-frame sign convention");
  std::cout << "  [ok] planner converges, fly_time=" << plan.timing.fly_time
            << " pitch=" << plan.aim.pitch << '\n';
}

// 弹速低于 14 m/s 时回退到 23；14 m/s 及以上不设上限。
void testPlannerUsesOneSidedBulletFallback()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 0.0;

  const auto plan = planner.plan(makeTarget(0.0), robot_state, {});
  require(plan.valid(), "fallback speed must still produce a plan");
  require(!plan.fireAdmissible(), "bad bullet speed must block firing");
  require(
    plan.reason == L4Planning::PlanError::BadBulletSpeed,
    "bad bullet speed must be reported");

  robot_state.bullet_speed = 13.99;
  const auto below = planner.plan(makeTarget(0.0), robot_state, {});
  require(below.valid(), "fallback must still yield aim angles");
  require(
    below.reason == L4Planning::PlanError::BadBulletSpeed,
    "speed below 14 m/s must use the configured fallback");

  for (const double speed : {14.0, 15.0, 45.0}) {
    robot_state.bullet_speed = speed;
    const auto accepted = planner.plan(makeTarget(0.0), robot_state, {});
    require(accepted.valid(), "planner must accept every speed at or above 14 m/s");
    require(
      accepted.reason == L4Planning::PlanError::None,
      "planner must not impose an upper bullet-speed gate");
  }
  std::cout << "  [ok] planner uses a one-sided 14 m/s fallback\n";
}

void testPlannerRejectsNoTarget()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  // 目标丢失由 Tracker 表达成"不返回目标"，所以规划器这一侧只剩空值这一种
  // 情况——不再有携带 Lost 状态的目标快照。
  const auto empty = planner.plan(std::nullopt, robot_state, {});
  require(!empty.valid(), "missing target must not produce a plan");
  require(empty.reason == L4Planning::PlanError::NoTarget, "NoTarget expected");

  // "滤波器为空的目标"不再是一种可表示的状态：TrackedTarget 没有默认构造，
  // 一经存在状态就是完整的十一维，所以这里只剩空值这一条拒绝路径。
  std::cout << "  [ok] planner rejects a missing target\n";
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
  require(blind.valid() && blind.fire.has_value(), "unobserved target must be trackable");
  require(blind.fire->armor_id == 0, "unobserved geometry must stay on armor 0");

  planner.reset();
  target.jumped = true;
  const auto seen = planner.plan(target, robot_state, {});
  require(seen.valid() && seen.fire.has_value(), "observed target must be plannable");
  require(seen.fire->armor_id == 1, "observed geometry must be free to pick armor 1");
  std::cout << "  [ok] unobserved geometry pins the aim to armor 0\n";
}

// 两块板同时留在 ±60° 窗口内时保持旧锁；只有旧板离开窗口才切到新板。
void testSelectorHoldsUntilArmorLeavesWindow()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  const auto degrees = [](double value) {
    return value * std::numbers::pi / 180.0;
  };

  const auto first = planner.plan(
    makeTarget(0.0, degrees(44.0)), robot_state, {});
  require(
    first.valid() && first.fire && first.fire->armor_id == 0,
    "selector did not initially lock armor 0");

  // 无目标和显式 reset 都不改变当前锁定板。
  const auto no_target = planner.plan(std::nullopt, robot_state, {});
  require(!no_target.valid(), "missing target must remain invalid");
  planner.reset();

  for (const double yaw_degrees : {50.0, 59.0}) {
    const auto plan = planner.plan(
      makeTarget(0.0, degrees(yaw_degrees)), robot_state, {});
    require(plan.valid() && plan.fire, "overlap-window plan must stay valid");
    require(plan.fire->armor_id == 0, "lock changed across no-target/reset or overlap");
  }

  const auto after_leaving = planner.plan(
    makeTarget(0.0, degrees(61.0)), robot_state, {});
  require(after_leaving.valid() && after_leaving.fire, "plan after leaving must stay valid");
  require(after_leaving.fire->armor_id == 3, "lock did not switch after armor 0 left");

  const auto overlap_again = planner.plan(
    makeTarget(0.0, degrees(59.0)), robot_state, {});
  require(overlap_again.valid() && overlap_again.fire, "returning overlap must stay valid");
  require(overlap_again.fire->armor_id == 3, "new lock was not retained in overlap");
  std::cout << "  [ok] selector holds a plate until it leaves the 60 deg window\n";
}

// 面对高速旋转的普通四板车，规划结果仍必须落在实体装甲板上。
void testPlannerAlwaysAimsAtPhysicalArmor()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  for (int step = 0; step < 60; ++step) {
    const auto target = makeTarget(
      20.0, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 60.0));
    const auto plan = planner.plan(target, robot_state, {});
    require(plan.valid(), "normal-car branch must keep producing an aim point");
    require(
      plan.fire && plan.fire->armor_id >= 0 && plan.fire->armor_id < 4 &&
        (plan.aim.point - plan.fire->point()).norm() < 1e-12,
      "selected armor id must be physical");
  }
  std::cout << "  [ok] high-speed path always aims at a physical armor\n";
}

void testSignedSpeedDelaySelection()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  const auto low = planner.plan(makeTarget(8.0), robot_state, {}, false);
  const auto high = planner.plan(makeTarget(8.01), robot_state, {}, false);
  const auto negative = planner.plan(makeTarget(-20.0), robot_state, {}, false);
  require(low.valid() && high.valid() && negative.valid(), "delay test plans must be valid");
  require(
    std::abs(low.timing.delay.beforeFire() - 0.020) < 1e-12,
    "w == decision_speed must use low delay");
  require(
    std::abs(high.timing.delay.beforeFire() - 0.035) < 1e-12,
    "positive w above decision_speed must use high delay");
  require(
    std::abs(negative.timing.delay.beforeFire() - 0.020) < 1e-12,
    "negative high speed must still use low delay");

  const auto target = makeTarget(0.0);
  const auto to_now = planner.plan(
    target, robot_state, target.t() + std::chrono::milliseconds(12), true);
  require(to_now.valid(), "to_now delay test plan must be valid");
  require(
    std::abs(to_now.timing.delay.beforeFire() - 0.027) < 1e-12,
    "to_now must add measured elapsed time to low delay");
  std::cout << "  [ok] signed speed delay and offline 5 ms path match\n";
}

// 每轮在共同命中时刻重新选板；扫描普通四板车确认迭代始终输出有限角度。
void testSharedIterationProducesFiniteCommands()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  for (const double v_yaw : {0.5, 2.0, 6.0, 12.0}) {
    for (int step = 0; step < 120; ++step) {
      const auto target = makeTarget(
        v_yaw, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 120.0));
      const auto plan = planner.plan(target, robot_state, {});
      require(plan.valid(), "shared iteration must yield a command here");
      require(
        std::isfinite(plan.aim.yaw) && std::isfinite(plan.aim.pitch),
        "angles must be finite");
    }
    planner.reset();
  }
  std::cout << "  [ok] shared predict/choose iteration is finite on 480 configurations\n";
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
  testPlannerConverges();
  testPlannerUsesOneSidedBulletFallback();
  testPlannerRejectsNoTarget();
  testUnobservedGeometryLocksArmorZero();
  testSelectorHoldsUntilArmorLeavesWindow();
  testPlannerAlwaysAimsAtPhysicalArmor();
  testSignedSpeedDelaySelection();
  testSharedIterationProducesFiniteCommands();
  std::cout << "planner smoke test passed\n";
  return 0;
}
