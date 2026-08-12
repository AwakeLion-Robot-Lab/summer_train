// L4 规划链路的行为冒烟测试：外推、弹道反解、以及 Planner 的延迟档、锁板和
// 命中时刻迭代。不依赖相机、串口和推理后端。

#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/planner.hpp"
#include "l4_planning/predictor.hpp"
#include "l3_estimation/filter_est/target.hpp"
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

// 正向弹道：给定发射角，算子弹飞到该水平距离时的高度和用时。测试用它验证反解。
struct Impact {
  double z{0.0};
  double fly_time{0.0};
};

Impact forwardImpact(double distance, double pitch, double v0, double gravity, double drag)
{
  const double range = drag < 1e-6 ? distance : std::expm1(drag * distance) / drag;
  const double fly_time = range / (v0 * std::cos(pitch));
  return {v0 * std::sin(pitch) * fly_time - 0.5 * gravity * fly_time * fly_time, fly_time};
}

void testPredictorAdvancesYaw()
{
  const L4Planning::Predictor predictor;
  const auto target = makeTarget(10.0);

  const auto later = predictor.predict(target, 0.1);
  require(
    std::abs(later.state()[L3Estimation::Yaw] - 1.0) < 1e-9,
    "yaw must advance by v_yaw * dt");

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
  L3Estimation::FilterEst::Target filter_target(observation, t0, 0.2, 4, P0);

  // 十帧，每帧 20 ms 沿 +x 前进 2 cm，即 1 m/s。
  for (int step = 1; step <= 10; ++step) {
    observation.xyz_in_world.x() = 3.8 + 0.02 * step;
    observation.ypd_in_world = L6Telemetry::xyz2ypd(observation.xyz_in_world);
    filter_target.predict(t0 + std::chrono::milliseconds(20 * step));
    filter_target.update(observation);
  }
  const L3Estimation::TrackedTarget target = filter_target.snapshot();
  require(
    target.state()[L3Estimation::VelocityX] > 0.1,
    "EKF must pick up a positive x velocity");

  const L3Estimation::TargetStateVector before = target.state();
  const auto later = predictor.predict(target, 0.5);
  const L3Estimation::TargetStateVector after = later.state();
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
  const L4Planning::BallisticConfig config;
  const double v0 = 23.0;

  for (const double distance : {1.5, 4.0, 7.0}) {
    for (const double height : {-0.5, 0.0, 0.8}) {
      const auto result = L4Planning::solveBallistic(distance, height, v0, config);
      require(result.has_value(), "vacuum ballistic must be solvable at short range");

      const Impact impact =
        forwardImpact(distance, result->pitch, v0, config.gravity, config.drag_coefficient);
      require(
        std::abs(impact.z - height) < 1e-6,
        "solved pitch must reproduce the target height");
      require(
        std::abs(result->fly_time - impact.fly_time) < 1e-9,
        "fly time must match the horizontal component");
    }
  }
  std::cout << "  [ok] vacuum ballistic round-trips\n";
}

// 阻力系数为 0 时必须严格退化成真空模型，否则默认配置一改就会引入静默偏差。
void testDragDegradesToVacuum()
{
  L4Planning::BallisticConfig vacuum;
  L4Planning::BallisticConfig zero_drag;
  zero_drag.drag_coefficient = 0.0;

  for (const double height : {-0.3, 0.0, 0.2}) {
    const auto a = L4Planning::solveBallistic(5.0, height, 23.0, vacuum);
    const auto b = L4Planning::solveBallistic(5.0, height, 23.0, zero_drag);
    require(a.has_value() && b.has_value(), "both configs must solve");
    require(std::abs(a->pitch - b->pitch) < 1e-12, "zero drag must match vacuum pitch");
    require(std::abs(a->fly_time - b->fly_time) < 1e-12, "zero drag must match vacuum time");
  }
  std::cout << "  [ok] quadratic drag degrades to vacuum at k=0\n";
}

// 有阻力时子弹更慢、掉得更多，所以必须抬得更高、飞得更久。
void testDragNeedsMorePitch()
{
  L4Planning::BallisticConfig config;
  const auto vacuum = L4Planning::solveBallistic(6.0, 0.0, 23.0, config);
  config.drag_coefficient = 0.02;
  const auto drag = L4Planning::solveBallistic(6.0, 0.0, 23.0, config);

  require(vacuum.has_value() && drag.has_value(), "both configs must solve at 6 m");
  require(drag->pitch > vacuum->pitch, "drag must require a higher muzzle angle");
  require(drag->fly_time > vacuum->fly_time, "drag must lengthen the flight");
  std::cout << "  [ok] quadratic drag raises pitch by "
            << (drag->pitch - vacuum->pitch) * 57.3 << " deg at 6 m\n";
}

// 反解是精确闭式解而不是迭代近似：代回正向模型的残差必须落在机器精度。
void testDragInverseIsExact()
{
  for (const double k : {0.0, 0.019, 0.092}) {
    L4Planning::BallisticConfig config;
    config.drag_coefficient = k;

    for (const double distance : {1.5, 4.0, 6.0, 10.0}) {
      for (const double height : {-0.4, 0.0, 0.6}) {
        const auto result = L4Planning::solveBallistic(distance, height, 23.0, config);
        require(result.has_value(), "closed form must solve inside the envelope");

        const Impact impact =
          forwardImpact(distance, result->pitch, 23.0, config.gravity, k);
        require(
          std::abs(impact.z - height) < 1e-9,
          "closed-form inverse must be exact, not merely within tolerance");
        require(
          std::abs(impact.fly_time - result->fly_time) < 1e-12,
          "fly time must agree between forward and inverse");
      }
    }
  }
  std::cout << "  [ok] drag inverse is exact to 1e-9 m at k = 0 / 0.019 / 0.092\n";
}

void testBallisticRejectsBadInput()
{
  const L4Planning::BallisticConfig config;
  const auto solve = [&config](double d, double h, double v0) {
    return L4Planning::solveBallistic(d, h, v0, config);
  };

  require(!solve(4.0, 0.0, 0.0).has_value(), "zero bullet speed must be rejected");
  require(!solve(0.0, 0.0, 23.0).has_value(), "zero distance must be rejected");
  require(!solve(500.0, 0.0, 23.0).has_value(), "out-of-range target must be rejected");
  require(!solve(std::nan(""), 0.0, 23.0).has_value(), "NaN distance must be rejected");

  // 业务门限全部收归 PlanConfig，求解器不得再自带一道弹速门槛，否则落在两道
  // 门限夹缝里的弹速会被报成 BallisticFailed 而不是 BadBulletSpeed。
  require(
    solve(4.0, 0.0, 15.0).has_value(),
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

  const auto plan = planner.plan(target, robot_state, target.timestamp());
  require(plan.valid, "static target must be plannable");
  require(plan.error == L4Planning::PlanError::None, "no rejection expected");
  require(plan.fire_admissible, "a static facing target must be shootable");
  require(plan.ballistic_valid, "ballistic must be valid");
  require(plan.fly_time > 0.0, "fly time must be positive");
  require(
    std::abs(plan.delay.fire_to_hit - plan.fly_time) < 1e-12,
    "fire_to_hit must carry the fly time");
  require(plan.hit_time > plan.fire_time, "hit must come after fire");
  require(plan.armor_id >= 0, "a physical armor must be selected");

  // 目标在 x 轴正方向，选中的板朝向枪口，yaw 应当接近 0。
  require(std::abs(plan.yaw) < 0.1, "yaw should point at the target");
  require(plan.pitch < 0.0, "command pitch must negate the ballistic pitch");
  std::cout << "  [ok] planner converges, fly_time=" << plan.fly_time
            << " pitch=" << plan.pitch << '\n';
}

// 弹速低于门限时仍然给瞄准角，但这一帧不允许开火；门限之上不设上限。
void testPlannerBulletSpeedFallback()
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
  require(below.valid, "fallback must still yield aim angles");
  require(
    below.error == L4Planning::PlanError::BadBulletSpeed,
    "speed below the threshold must use the fallback");

  for (const double speed : {14.0, 15.0, 45.0}) {
    robot_state.bullet_speed = speed;
    const auto accepted = planner.plan(makeTarget(0.0), robot_state, {});
    require(accepted.valid, "every speed at or above the threshold must be accepted");
    require(
      accepted.error == L4Planning::PlanError::None,
      "there must be no upper bullet-speed gate");
  }
  std::cout << "  [ok] planner falls back below 14 m/s and blocks firing\n";
}

void testPlannerRejectsNoTarget()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  // 目标丢失由 Tracker 表达成"不返回目标"，所以规划器这一侧只剩空值这一种情况。
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

  target.jumped = true;
  const auto seen = planner.plan(target, robot_state, {});
  require(seen.valid, "observed target must be plannable");
  require(seen.armor_id == 1, "observed geometry must be free to pick armor 1");
  std::cout << "  [ok] unobserved geometry pins the aim to armor 0\n";
}

// 锁板边界：两块板同时留在 ±60° 窗口内时，即使另一块已经更正对枪口也不能
// 提前换；只有旧板离开窗口才切到新板。
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
  require(first.valid && first.armor_id == 0, "armor 0 must be locked first");

  for (const double yaw_degrees : {50.0, 59.0}) {
    const auto plan = planner.plan(
      makeTarget(0.0, degrees(yaw_degrees)), robot_state, {});
    require(plan.valid, "overlap-window plan must stay valid");
    require(plan.armor_id == 0, "lock must hold while both plates overlap");
  }

  const auto after_leaving = planner.plan(
    makeTarget(0.0, degrees(61.0)), robot_state, {});
  require(after_leaving.valid, "plan after leaving the window must stay valid");
  require(after_leaving.armor_id == 3, "lock did not switch after armor 0 left");

  const auto overlap_again = planner.plan(
    makeTarget(0.0, degrees(59.0)), robot_state, {});
  require(overlap_again.valid, "returning overlap-window plan must stay valid");
  require(overlap_again.armor_id == 3, "the new lock was not retained in overlap");
  std::cout << "  [ok] selector holds a plate until it leaves the 60 deg window\n";
}

// 普通四板车即使转得很快也始终瞄实体装甲板，不会退化成中心代理点。
void testAlwaysAimsAtPhysicalArmor()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  for (int step = 0; step < 60; ++step) {
    const auto target = makeTarget(
      20.0, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 60.0));
    const auto plan = planner.plan(target, robot_state, {});
    require(plan.valid, "the normal-car branch must keep producing an aim point");
    require(
      plan.armor_id >= 0 && plan.armor_id < 4,
      "the selected armor id must be physical");
  }
  std::cout << "  [ok] high-speed path always aims at a physical armor\n";
}

void testDelaySelection()
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
    "w == decision_speed must use the low delay");
  require(
    std::abs(high.delay.beforeFire() - 0.035) < 1e-12,
    "positive w above decision_speed must use the high delay");
  // 比较是有符号的：反向高速旋转仍走低速档。
  require(
    std::abs(negative.delay.beforeFire() - 0.020) < 1e-12,
    "negative high speed must still use the low delay");

  const auto target = makeTarget(0.0);
  const auto to_now = planner.plan(
    target, robot_state, target.timestamp() + std::chrono::milliseconds(12), true);
  require(to_now.valid, "to_now delay test plan must be valid");
  require(
    std::abs(to_now.delay.beforeFire() - 0.027) < 1e-12,
    "to_now must add the measured elapsed time to the low delay");
  std::cout << "  [ok] signed speed delay and the offline 5 ms path match\n";
}

// 每轮迭代都在共同命中时刻重新选板；扫描普通四板车确认输出始终有限。
void testSharedIterationProducesFiniteCommands()
{
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  for (const double v_yaw : {0.5, 2.0, 6.0, 12.0}) {
    L4Planning::Planner planner;
    for (int step = 0; step < 120; ++step) {
      const auto target = makeTarget(
        v_yaw, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 120.0));
      const auto plan = planner.plan(target, robot_state, {});
      require(plan.valid, "the shared iteration must yield a command here");
      require(std::isfinite(plan.yaw) && std::isfinite(plan.pitch), "angles must be finite");
    }
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
  testBallisticRejectsBadInput();
  testPlannerConverges();
  testPlannerBulletSpeedFallback();
  testPlannerRejectsNoTarget();
  testUnobservedGeometryLocksArmorZero();
  testSelectorHoldsUntilArmorLeavesWindow();
  testAlwaysAimsAtPhysicalArmor();
  testDelaySelection();
  testSharedIterationProducesFiniteCommands();
  std::cout << "planner smoke test passed\n";
  return 0;
}
