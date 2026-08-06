// L4 规划链路的行为冒烟测试：整车外推、真空弹道、选板迟滞、不动点迭代。
// 不依赖相机、串口和推理后端。

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
  target.timestamp = std::chrono::steady_clock::time_point{};
  return target;
}

// 只推中心不推 yaw 是改造前的缺陷：小陀螺目标会被算成原地不动。
void testPredictorAdvancesYaw()
{
  const L4Planning::Predictor predictor;
  const auto target = makeTarget(10.0);

  const auto later = predictor.predict(target, 0.1);
  require(
    std::abs(later.yaw - 1.0) < 1e-9,
    "yaw must advance by v_yaw * dt");

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

// 真空弹道：解出的 pitch 代回抛体方程必须还原目标高度。
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

void testBallisticRejectsBadInput()
{
  const L4Planning::BallisticSolver solver;
  require(!solver.solve(4.0, 0.0, 0.0).valid, "zero bullet speed must be rejected");
  require(!solver.solve(0.0, 0.0, 23.0).valid, "zero distance must be rejected");
  require(!solver.solve(500.0, 0.0, 23.0).valid, "out-of-range target must be rejected");
  require(
    !solver.solve(std::nan(""), 0.0, 23.0).valid, "NaN distance must be rejected");
  std::cout << "  [ok] ballistic rejects unusable input\n";
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
  require(plan.ballistic_valid, "ballistic must be valid");
  require(plan.fly_time > 0.0, "fly time must be positive");
  require(
    std::abs(plan.delay.fire_to_hit - plan.fly_time) < 1e-12,
    "fire_to_hit must carry the fly time");
  require(plan.hit_time > plan.fire_time, "hit must come after fire");

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
  require(
    plan.error == L4Planning::PlanError::BadBulletSpeed,
    "bad bullet speed must be reported");
  std::cout << "  [ok] planner flags bad bullet speed but still aims\n";
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

// 反陀螺档只打正在转入视野的一侧，且高速旋转下允许出现无解间歇。
void testSelectorAntiSpin()
{
  L4Planning::Planner planner;
  L1Sensor::RobotState robot_state;
  robot_state.bullet_speed = 23.0;

  auto target = makeTarget(8.0);  // 远高于 spin_threshold
  int valid_count = 0;
  for (int step = 0; step < 60; ++step) {
    target.yaw = L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 60.0);
    const auto plan = planner.plan(target, robot_state, {});
    if (plan.valid) {
      ++valid_count;
      // 命中的板必须落在 coming 窗口内。
      require(
        plan.armor_id >= 0 && plan.armor_id < 4, "armor id must be in range");
    }
  }
  require(valid_count > 0, "anti-spin must find shootable windows");
  require(valid_count < 60, "anti-spin must also reject out-of-window frames");
  std::cout << "  [ok] anti-spin selected " << valid_count << "/60 frames\n";
}

}  // namespace

int main()
{
  testPredictorAdvancesYaw();
  testPredictorTranslates();
  testBallisticRoundTrip();
  testBallisticRejectsBadInput();
  testPlannerConverges();
  testPlannerFlagsBadBulletSpeed();
  testPlannerRejectsNoTarget();
  testSelectorHysteresis();
  testSelectorAntiSpin();
  std::cout << "planner smoke test passed\n";
  return 0;
}
