#include "l3_planning/armor_candidate_generator.hpp"
#include "l3_planning/in_plate_aim.hpp"
#include "l3_planning/planner.hpp"
#include "l3_planning/planner_config.hpp"
#include "l3_planning/tiny_mpc.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string_view>

using namespace L4Planning;
using namespace std::chrono_literals;
namespace {
int failures = 0;
void expect(bool condition, std::string_view message)
{
  if (!condition) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
double wrap(double x) { return std::remainder(x, 2.0 * std::numbers::pi); }
L3Estimation::TargetState targetAt(double rate, int count = 4)
{
  L3Estimation::TargetState target;
  target.robot_id = count == 3 ? 6 : 3;
  target.armor_count = count;
  target.radius = 0.26;
  target.height_offset = 0.04;
  target.three_armor_height_offsets = {0.0, 0.01, 0.04};
  target.center = {4.0, 0.0, 0.2};
  target.yaw_rate = rate;
  target.timestamp = TimePoint{2s};
  target.covariance.setZero();
  return target;
}
InPlateAimState stateAt(const L3Estimation::TargetState& target, int id)
{
  InPlateAimState state;
  state.robot_id = target.robot_id;
  state.armor_id = id;
  state.stable_frames = 10;
  state.timestamp = target.timestamp - 1s;
  return state;
}
void directionChecks()
{
  InPlateAimConfig config;
  config.enable = true;
  for (const int count : {3, 4}) {
    for (const double rate : {3.0, -3.0}) {
      for (const double bearing : {0.0, 1.4, 3.13, -3.13}) {
        for (int id = 0; id < count; ++id) {
          auto target = targetAt(rate, count);
          target.center.x() = 4.0 * std::cos(bearing);
          target.center.y() = 4.0 * std::sin(bearing);
          target.yaw = wrap(bearing - id * 2.0 * std::numbers::pi / count);
          const auto prediction = Predictor{}.predict({target, target.timestamp});
          const auto& armor = prediction.armor_candidates[id];
          const double gun_yaw = std::atan2(armor.position_world.y(), armor.position_world.x());
          const auto aim = queryInPlateAim(prediction, armor, stateAt(target, id), target.timestamp, gun_yaw, config);
          expect(aim.valid && aim.offset.x() * rate > 0.0,
            "horizontal offset opposes rotational center motion in either direction and across yaw wrap");
          const int next = nextArmorInRotation(id, count, rate > 0.0 ? 1 : -1);
          const double dt = 2.0 * std::numbers::pi / count / std::abs(rate);
          const auto at_next = Predictor{}.predict(target, dt);
          const auto next_prediction = Predictor{}.predict({at_next, at_next.timestamp});
          expect(std::abs(wrap(next_prediction.armor_candidates[next].yaw_world - armor.yaw_world)) < 1e-9,
            "successor ID matches physical rotation after 90/120 degrees");
        }
      }
    }
  }
  for (const int old_direction : {1, -1}) {
    auto target = targetAt(-old_direction * 0.10);
    const auto prediction = Predictor{}.predict({target, target.timestamp});
    auto state = stateAt(target, 0);
    state.direction = old_direction;
    const auto aim = queryInPlateAim(prediction, prediction.armor_candidates.front(), state, target.timestamp, 0.0, config);
    expect(aim.direction == 0 && aim.next_armor_id == -1,
      "unconfirmed reversal stops directional pre-aim instead of choosing the old successor");
  }
}
void heightChecks()
{
  InPlateAimConfig config;
  config.enable = true;
  for (const double rate : {0.5, -0.5, 1.0, -1.0, 4.0, -4.0, 8.0, -8.0}) {
    double previous = 0.0;
    for (int id = 0; id < 4; ++id) {
      auto target = targetAt(rate);
      target.yaw = -id * std::numbers::pi / 2.0;
      const auto prediction = Predictor{}.predict({target, target.timestamp});
      const auto& armor = prediction.armor_candidates[id];
      const auto aim = queryInPlateAim(prediction, armor, stateAt(target, id), target.timestamp, 0.0, config);
      expect(std::abs(aim.position_world.z() - 0.22) < 1e-4,
        "overlapping high/low plates converge to common height at slow and fast spin");
      if (id > 0) expect(std::abs(aim.position_world.z() - previous) < 1e-4,
        "high/low shared height is symmetric across plate handoffs");
      previous = aim.position_world.z();
    }
  }
  auto target = targetAt(4.0);
  target.yaw = -std::numbers::pi / 2.0; // 当前板 1，上一板 0。
  target.center.z() = 0.30; // 车体从上次快照时刻上移了 10 cm。
  const auto prediction = Predictor{}.predict({target, target.timestamp});
  auto state = stateAt(target, 0);
  state.timestamp = target.timestamp; // 只检查换板映射，不推进偏移运动。
  state.offset.y() = 0.02;
  state.position_world = {3.7, 0.0, 0.20 + 0.02 * std::cos(std::numbers::pi / 12.0)};
  const auto& previous = prediction.armor_candidates[0];
  const double expected_height = previous.position_world.z()
    + previous.rotation_world(2, 1) * state.offset.x() + previous.rotation_world(2, 2) * state.offset.y();
  const auto switched = queryInPlateAim(prediction, prediction.armor_candidates[1], state, target.timestamp, 0.0, config);
  expect(std::abs(switched.position_world.z() - expected_height) < 1e-9,
    "handoff maps previous plate offset at current impact time instead of freezing stale world height");
}
void mpcRolloutChecks()
{
  for (const double rate : {6.0, -6.0}) {
    auto target = targetAt(rate);
    target.velocity.z() = 0.10;
    PlannerContext context;
    context.config = loadPlannerTuning("config/planner_config.yaml").planner;
    context.config.in_plate_aim.enable = true;
    context.config.in_plate_aim.mode = InPlateAimMode::Independent;
    L1Sensor::RobotState robot;
    robot.bullet_speed = 25.0;
    robot.timestamp = target.timestamp;
    auto state = stateAt(target, 0);
    state.timestamp = target.timestamp - 20ms;
    state.gun_yaw_rate = rate > 0.0 ? 0.25 : -0.25;
    state.offset = Eigen::Vector2d{0.005, 0.01};
    state.velocity = Eigen::Vector2d{0.02, -0.01};
    const auto fire_time = target.timestamp + 15ms;
    const auto candidates = generateArmorCandidates(target, robot, context, fire_time, state, 0);
    expect(candidates && !candidates->empty() && candidates->front().valid, "rollout direct candidate valid");
    if (!candidates || candidates->empty()) continue;
    const auto& candidate = candidates->front();
    AimPlan plan;
    plan.valid = plan.armor_geometry_valid = true;
    plan.armor_id = 0;
    plan.fire_time = fire_time;
    plan.impact_time = candidate.impact_time;
    plan.fly_time = candidate.ballistic.fly_time;
    plan.yaw = candidate.ballistic.yaw;
    plan.pitch = candidate.ballistic.pitch;
    plan.aim_query_state = state;
    const auto reference = buildTinyMpcReference(plan, target, robot, Eigen::Isometry3d::Identity(), context.config);
    expect(reference.valid, "MPC rollout reference valid");
    if (!reference.valid) continue;
    expect((reference.in_plate_aim[kTinyMpcControlIndex].offset - candidate.aim.offset).norm() < 1e-12,
      "rollout keeps exact direct-plan center");
    auto local = state;
    local.robot_id = target.robot_id;
    local.armor_id = 0;
    local.timestamp = fire_time;
    local.offset = candidate.aim.offset;
    local.velocity = candidate.aim.velocity;
    local.direction = candidate.aim.direction;
    local.position_world = candidate.aim.position_world;
    int mismatches = 0;
    int switches = 0;
    for (std::size_t i = kTinyMpcControlIndex + 1; i < kTinyMpcHorizon; ++i) {
      const auto time = fire_time + std::chrono::milliseconds(10 * (i - kTinyMpcControlIndex));
      auto sample_robot = robot;
      sample_robot.rpy.yaw = wrap(robot.rpy.yaw + state.gun_yaw_rate
        * 0.01 * (i - kTinyMpcControlIndex));
      const auto expected = generateArmorCandidates(target, sample_robot, context, time, local, reference.armor_id[i]);
      if (!expected || expected->empty() || !expected->front().valid) { ++mismatches; continue; }
      const auto& point = expected->front().aim;
      if ((point.offset - reference.in_plate_aim[i].offset).norm() > 1e-9
          || (point.velocity - reference.in_plate_aim[i].velocity).norm() > 1e-9) ++mismatches;
      if (local.armor_id != reference.armor_id[i]) {
        expect(reference.armor_id[i] == nextArmorInRotation(local.armor_id, 4, rate > 0 ? 1 : -1),
          "four-plate MPC switches in physical rotation order");
        ++switches;
      }
      local.armor_id = reference.armor_id[i];
      local.timestamp = time;
      local.offset = reference.in_plate_aim[i].offset;
      local.velocity = reference.in_plate_aim[i].velocity;
      local.direction = reference.in_plate_aim[i].direction;
      local.position_world = reference.in_plate_aim[i].position_world;
    }
    if (mismatches) std::cerr << "MPC rollout mismatches=" << mismatches << " at rate=" << rate << '\n';
    expect(mismatches == 0 && switches >= 1, "future samples advance selected local aim state once per step through handoff");
    expect(plan.aim_query_state.timestamp == state.timestamp && plan.aim_query_state.offset == state.offset,
      "MPC rollout does not commit cross-frame planner state");
  }
}

void trackingGapChecks()
{
  auto config = loadPlannerTuning("config/planner_config.yaml").planner;
  config.in_plate_aim.enable = true;
  config.in_plate_aim.mode = InPlateAimMode::Independent;
  config.enable_mpc = false;
  config.lock_stable_frames = 1;
  config.switch_yaw_dead_zone = config.switch_pitch_dead_zone = 90.0;
  for (const double rate : {6.0, -6.0}) {
    Planner planner(config);
    auto target = targetAt(rate);
    L1Sensor::RobotState robot;
    robot.bullet_speed = 25.0;
    int invalid_frames = 0;
    bool started = false;
    int last_armor = -1;
    int switches = 0;
    for (int frame = 0; frame < 240; ++frame) {
      robot.timestamp = target.timestamp;
      const auto plan = planner.plan(target, robot);
      if (!plan.valid) {
        if (started) ++invalid_frames;
      } else {
        if (last_armor >= 0 && plan.armor_id != last_armor) ++switches;
        last_armor = plan.armor_id;
        if (!plan.within_firing_window) expect(!plan.fire_permitted, "selection gap never permits fire");
        if (started) expect(plan.aim_query_state.stable_frames > 0, "window gap does not reset in-plate warmup");
        started = true;
        robot.rpy.yaw = plan.yaw;
        robot.rpy.pitch = -plan.pitch;
      }
      target = Predictor{}.predict(target, 0.01);
    }
    if (invalid_frames) std::cerr << "tracking gap invalid frames=" << invalid_frames << " at rate=" << rate << '\n';
    expect(invalid_frames == 0 && switches >= 4, "continuous spin tracks across selection gaps without resetting aim state");
  }
}

}
int main()
{
  directionChecks();
  heightChecks();
  trackingGapChecks();
  mpcRolloutChecks();
  if (failures == 0) std::cout << "dynamic_aim_audit_smoke: all checks passed\n";
  return failures == 0 ? 0 : 1;
}
