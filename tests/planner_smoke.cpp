// L4 规划链路的行为冒烟测试：覆盖预测、弹道、延迟、锁板和命中时刻迭代。
// 不依赖相机、串口和推理后端。

#include "l4_planning/armor/planner.hpp"
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
L3Estimation::EskfTarget makeTarget(double v_yaw, double yaw = 0.0)
{
  L3Estimation::EskfTarget target(
    L3Estimation::ArmorName::Infantry3, 4.0, v_yaw, 0.2, yaw);
  target.jumped = true;
  return target;
}

// 常用输入：弹速 23 m/s；to_now = true 时 plan_time 取默认的零点。
L4Planning::PlanInput makeInput(
  const std::optional<L3Estimation::EskfTarget>& target, bool to_now = true)
{
  return {.target = target, .bullet_speed = 23.0, .to_now = to_now};
}

// 只推中心不推 yaw 是改造前的缺陷：小陀螺目标会被算成原地不动。规划器靠
// EskfTarget::predict 外推，这里钉住它同时推进 yaw。
void testPredictAdvancesYaw()
{
  const auto target = makeTarget(10.0);
  auto later = target;
  later.predict(0.1);
  require(
    std::abs(later.ekf_x()[6] - 1.0) < 1e-9, "yaw must advance by v_yaw * dt");

  const auto before = target.armor_xyza_list();
  const auto after = later.armor_xyza_list();
  require(before.size() == 4 && after.size() == 4, "four armor plates expected");

  const double moved = (before[0].head<3>() - after[0].head<3>()).norm();
  require(moved > 0.1, "spinning target armor must move over 100 ms");
  std::cout << "  [ok] predict advances yaw, armor moved " << moved << " m\n";
}

// 平动目标：中心按速度走，装甲板整体跟着平移。平动速度只能由 EKF 从观测里估
// 出来——确定性构造入口给不出 vx，所以这里先喂一段匀速直线运动的观测。
void testPredictTranslates()
{
  const std::chrono::steady_clock::time_point t0{};

  // 直接构造一个已在运动的目标：旋转中心 (4.0, 0, 0)，沿 +x 以 1 m/s 前进。
  // 不再靠喂十帧观测把速度攒出来——端点观测需要相机标定和投影，那是 L3 自己
  // 单测的事（tests/eskf_target_smoke.cpp），规划层这里只需要一个会动的目标。
  L3Estimation::EskfTarget target(
    L3Estimation::ArmorName::Infantry3, 4.0, 0.0, 0.2, 0.0, 0.0,
    Eigen::Vector3d{1.0, 0.0, 0.0});
  target.predict(t0);
  require(target.ekf_x()[1] > 0.1, "target must carry a positive x velocity");

  const Eigen::VectorXd before = target.ekf_x();
  auto later = target;
  later.predict(0.5);
  const Eigen::VectorXd after = later.ekf_x();
  require(
    std::abs(after[0] - (before[0] + before[1] * 0.5)) < 1e-9,
    "center must translate by velocity * dt");
  require(
    std::abs(
      after[6] - L6Telemetry::limit_rad(before[6] + before[7] * 0.5)) < 1e-9,
    "yaw must advance by exactly v_yaw * dt");
  std::cout << "  [ok] predict translates a target at "
            << before[1] << " m/s\n";
}

// 真空弹道：规划出的 pitch 代回抛体方程必须还原命中点高度。
void testBallisticRoundTrip()
{
  constexpr double kGravity = 9.7833;
  const double v0 = 23.0;
  L4Planning::Planner planner;

  for (const double center_x : {1.5, 4.0, 7.0}) {
    L3Estimation::EskfTarget target(
      L3Estimation::ArmorName::Infantry3, center_x, 0.0, 0.2);
    target.jumped = true;
    const auto plan = planner.plan(makeInput(target));
    require(plan.valid(), "vacuum ballistic must be solvable at short range");

    // 世界系 pitch 向下为正，弹道仰角是它的相反数。
    const double pitch = -plan.aim.pitch;
    const double distance = std::hypot(plan.aim.point.x(), plan.aim.point.y());
    // z = d*tan(theta) - g*d² / (2*v0²*cos²theta)
    const double cos_pitch = std::cos(pitch);
    const double reconstructed =
      distance * std::tan(pitch) -
      kGravity * distance * distance / (2.0 * v0 * v0 * cos_pitch * cos_pitch);
    require(
      std::abs(reconstructed - plan.aim.point.z()) < 1e-6,
      "solved pitch must reproduce the target height");
    require(
      std::abs(plan.timing.fly_time - distance / (v0 * cos_pitch)) < 1e-9,
      "fly time must match the horizontal component");
  }
  std::cout << "  [ok] vacuum ballistic round-trips\n";
}

// 不动点迭代应当收敛，且命中时刻的瞄准点确实由飞行时间决定。
void testPlannerConverges()
{
  L4Planning::Planner planner;
  const auto target = makeTarget(0.0);

  auto input = makeInput(target);
  input.plan_time = target.t();
  const auto plan = planner.plan(input);
  require(plan.valid(), "static target must be plannable");
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

// 电控没发弹速（0）或低于 14 m/s 时按缺省 23 m/s 解弹道，计划照常有效、照常
// 可开火；14 m/s 及以上不设上限。
void testPlannerDefaultsBulletSpeed()
{
  L4Planning::Planner planner;
  auto input = makeInput(makeTarget(0.0));
  const auto reference = planner.plan(input);

  for (const double speed : {0.0, 13.99, std::nan("")}) {
    input.bullet_speed = speed;
    const auto plan = planner.plan(input);
    require(plan.valid(), "an unusable bullet speed must still produce a fireable plan");
    require(
      std::abs(plan.timing.fly_time - reference.timing.fly_time) < 1e-12,
      "an unusable bullet speed must be replaced by the 23 m/s default");
  }

  for (const double speed : {14.0, 15.0, 45.0}) {
    input.bullet_speed = speed;
    const auto accepted = planner.plan(input);
    require(accepted.valid(), "planner must accept every speed at or above 14 m/s");
    require(
      std::abs(accepted.timing.fly_time - reference.timing.fly_time) > 1e-6 ||
        speed == 23.0,
      "a valid MCU speed must be used as-is");
  }
  std::cout << "  [ok] planner defaults to 23 m/s below 14 m/s and keeps firing\n";
}

// 五段延迟里，runtime 实测的两段必须真的进到 beforeFire()，而不是恒为 0。
void testDelayChainCarriesEveryStage()
{
  L4Planning::PlanConfig config;
  config.send_to_control = 0.004;
  L4Planning::Planner planner(config);

  // to_now = false：image_to_plan 走离线的固定 5 ms
  auto input = makeInput(makeTarget(0.0), false);
  input.plan_to_send = 0.003;

  const auto plan = planner.plan(input);
  const auto & delay = plan.timing.delay;
  require(std::abs(delay.image_to_plan - 0.005) < 1e-12, "image_to_plan is wrong");
  require(std::abs(delay.plan_to_send - 0.003) < 1e-12, "plan_to_send was not carried");
  require(std::abs(delay.send_to_control - 0.004) < 1e-12, "send_to_control was not carried");
  require(delay.control_to_fire > 0.0, "control_to_fire must come from the speed bands");
  require(std::abs(delay.fire_to_hit - plan.timing.fly_time) < 1e-12, "fire_to_hit is wrong");
  require(
    std::abs(delay.beforeFire() -
             (delay.image_to_plan + delay.plan_to_send + delay.send_to_control +
              delay.control_to_fire)) < 1e-12,
    "beforeFire must sum every stage before the shot");
  std::cout << "  [ok] delay chain carries every stage, before_fire="
            << delay.beforeFire() << " s\n";
}

// coming_angle 以前在常规车那条分支里是写死的 60 度，配置改了不生效。
void testComingAngleIsConfigurable()
{
  // 让 0 号板偏离视线 50 度：默认 60 度窗口收得下，收紧到 40 度就该落空。
  const auto input = makeInput(makeTarget(0.0, 50.0 / 57.3));

  L4Planning::Planner wide;
  require(
    wide.plan(input).fire.has_value(),
    "a 50-degree armor must fit inside the default 60-degree window");

  L4Planning::PlanConfig narrow_config;
  narrow_config.coming_angle = 40.0 / 57.3;
  L4Planning::Planner narrow(narrow_config);
  const auto narrowed = narrow.plan(input);
  require(
    narrowed.error == L4Planning::PlanError::OutOfWindow,
    "a tightened coming_angle must actually shrink the normal-branch window");
  std::cout << "  [ok] coming_angle drives the normal branch\n";
}

void testPlannerRejectsNoTarget()
{
  L4Planning::Planner planner;

  // 目标丢失由 Tracker 表达成"不返回目标"，所以规划器这一侧只剩空值这一种
  // 情况——不再有携带 Lost 状态的目标快照。
  const auto empty = planner.plan(makeInput(std::nullopt));
  require(!empty.valid(), "missing target must not produce a plan");
  require(empty.error == L4Planning::PlanError::NoTarget, "NoTarget expected");

  // "滤波器为空的目标"不再是一种可表示的状态：EskfTarget 没有默认构造，
  // 一经存在状态就是完整的，所以这里只剩空值这一条拒绝路径。
  std::cout << "  [ok] planner rejects a missing target\n";
}

// 只见过一块板时整车 yaw、第二组半径、高度差都还没被观测约束过，瞄别的板
// 等于拿伪造的几何开火。
void testUnobservedGeometryLocksArmorZero()
{
  L4Planning::Planner planner;

  // 把整车转到 1 号板正对枪口的姿态：几何可观测时会选 1 号。
  auto target = makeTarget(0.0, -std::numbers::pi / 2.0);
  target.jumped = false;

  const auto blind = planner.plan(makeInput(target));
  require(blind.valid() && blind.fire.has_value(), "unobserved target must be trackable");
  require(blind.fire->armor_id == 0, "unobserved geometry must stay on armor 0");

  planner.reset();
  target.jumped = true;
  const auto seen = planner.plan(makeInput(target));
  require(seen.valid() && seen.fire.has_value(), "observed target must be plannable");
  require(seen.fire->armor_id == 1, "observed geometry must be free to pick armor 1");
  std::cout << "  [ok] unobserved geometry pins the aim to armor 0\n";
}

// 两块板同时留在 ±60° 窗口内时保持旧锁；只有旧板离开窗口才切到新板。
void testSelectorHoldsUntilArmorLeavesWindow()
{
  L4Planning::Planner planner;

  const auto degrees = [](double value) {
    return value * std::numbers::pi / 180.0;
  };

  const auto first = planner.plan(makeInput(makeTarget(0.0, degrees(44.0))));
  require(
    first.valid() && first.fire && first.fire->armor_id == 0,
    "selector did not initially lock armor 0");

  // 无目标和显式 reset 都不改变当前锁定板。
  const auto no_target = planner.plan(makeInput(std::nullopt));
  require(!no_target.valid(), "missing target must remain invalid");
  planner.reset();

  for (const double yaw_degrees : {50.0, 59.0}) {
    const auto plan = planner.plan(makeInput(makeTarget(0.0, degrees(yaw_degrees))));
    require(plan.valid() && plan.fire, "overlap-window plan must stay valid");
    require(plan.fire->armor_id == 0, "lock changed across no-target/reset or overlap");
  }

  const auto after_leaving = planner.plan(makeInput(makeTarget(0.0, degrees(61.0))));
  require(after_leaving.valid() && after_leaving.fire, "plan after leaving must stay valid");
  require(after_leaving.fire->armor_id == 3, "lock did not switch after armor 0 left");

  const auto overlap_again = planner.plan(makeInput(makeTarget(0.0, degrees(59.0))));
  require(overlap_again.valid() && overlap_again.fire, "returning overlap must stay valid");
  require(overlap_again.fire->armor_id == 3, "new lock was not retained in overlap");
  std::cout << "  [ok] selector holds a plate until it leaves the 60 deg window\n";
}

// 进自瞄的头一次选板按枪口挑：0 号板更正对，但枪口正对 3 号板时先锁 3 号；
// 之后枪口怎么转都沿用锁，直到 reset 开始新一轮自瞄。
void testEntryPicksArmorNearestMuzzle()
{
  const auto target = makeTarget(0.0, 44.0 * std::numbers::pi / 180.0);
  const auto armors = target.armor_xyza_list();
  const auto aimAt = [&](int id) {
    return Eigen::Quaterniond::FromTwoVectors(
      Eigen::Vector3d::UnitX(),
      armors[static_cast<std::size_t>(id)].head<3>().normalized());
  };
  const auto planWith = [&](L4Planning::Planner& planner, int muzzle_id) {
    auto input = makeInput(target, false);
    input.q_world_barrel = aimAt(muzzle_id);
    return planner.plan(input);
  };

  L4Planning::Planner frontal;
  const auto usual = frontal.plan(makeInput(target, false));
  require(usual.valid() && usual.fire && usual.fire->armor_id == 0,
          "without a muzzle pose the more frontal armor 0 must win");

  L4Planning::Planner planner;
  // 无目标的帧不算进入完成，第一条真正发出去的命令才算。
  require(!planner.plan(makeInput(std::nullopt)).valid(), "no target must be rejected");
  const auto entry = planWith(planner, 3);
  require(entry.valid() && entry.fire && entry.fire->armor_id == 3,
          "entry must lock the armor nearest the muzzle");
  const auto held = planWith(planner, 0);
  require(held.valid() && held.fire && held.fire->armor_id == 3,
          "after entry the lock must not follow the muzzle");

  planner.reset();
  const auto again = planWith(planner, 0);
  require(again.valid() && again.fire && again.fire->armor_id == 0,
          "a new aim session must pick by muzzle again");
  std::cout << "  [ok] entry locks the armor nearest the muzzle, then holds\n";
}

// 面对高速旋转的普通四板车，规划结果仍必须落在实体装甲板上。
void testPlannerAlwaysAimsAtPhysicalArmor()
{
  L4Planning::Planner planner;

  for (int step = 0; step < 60; ++step) {
    const auto target = makeTarget(
      20.0, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 60.0));
    const auto plan = planner.plan(makeInput(target));
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

  const auto low = planner.plan(makeInput(makeTarget(8.0), false));
  const auto high = planner.plan(makeInput(makeTarget(8.01), false));
  const auto negative = planner.plan(makeInput(makeTarget(-20.0), false));
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
  auto input = makeInput(target);
  input.plan_time = target.t() + std::chrono::milliseconds(12);
  const auto to_now = planner.plan(input);
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

  for (const double v_yaw : {0.5, 2.0, 6.0, 12.0}) {
    for (int step = 0; step < 120; ++step) {
      const auto target = makeTarget(
        v_yaw, L6Telemetry::limit_rad(step * 2.0 * std::numbers::pi / 120.0));
      const auto plan = planner.plan(makeInput(target));
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
  testPredictAdvancesYaw();
  testPredictTranslates();
  testBallisticRoundTrip();
  testPlannerConverges();
  testPlannerDefaultsBulletSpeed();
  testDelayChainCarriesEveryStage();
  testComingAngleIsConfigurable();
  testPlannerRejectsNoTarget();
  testUnobservedGeometryLocksArmorZero();
  testSelectorHoldsUntilArmorLeavesWindow();
  testEntryPicksArmorNearestMuzzle();
  testPlannerAlwaysAimsAtPhysicalArmor();
  testSignedSpeedDelaySelection();
  testSharedIterationProducesFiniteCommands();
  std::cout << "planner smoke test passed\n";
  return 0;
}
