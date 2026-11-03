#include "l3_planning/in_plate_aim.hpp"
#include "l3_planning/armor_candidate_generator.hpp"
#include "l3_planning/planner.hpp"
#include "l3_planning/planner_config.hpp"
#include "l3_planning/tiny_mpc.hpp"
#include "l4_control/fire_decision.hpp"
#include "runtime/l4_target_adapter.hpp"

#include <Eigen/Geometry>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>

using namespace L4Planning;
using namespace std::chrono_literals;
namespace VM = L3Estimation::VehicleModel;
namespace {
int failures = 0;
void expect(bool ok, std::string_view message)
{
  if (!ok) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

L3Estimation::TargetState numericalTarget()
{
  L3Estimation::TargetState target;
  target.robot_id = 3;
  target.center = {4.0, 0.0, 0.2};
  target.radius = 0.26;
  target.yaw_rate = 2.0;
  target.height_offset = 0.04;
  target.timestamp = TimePoint{1s};
  target.covariance.setZero();
  return target;
}

void policyChecks()
{
  auto target = numericalTarget();
  Predictor predictor;
  auto prediction = predictor.predict({target, target.timestamp});
  auto armor = prediction.armor_candidates.front();
  InPlateAimConfig config;
  config.enable = true;
  InPlateAimState state;
  state.robot_id = target.robot_id;
  state.armor_id = 0;
  state.timestamp = target.timestamp;
  state.stable_frames = 10;
  const auto time = target.timestamp + 100ms;
  const auto point = queryInPlateAim(prediction, armor, state, time, 0.02, config);
  const auto repeated = queryInPlateAim(prediction, armor, state, time, 0.02, config);
  expect(point.valid && point.offset.x() > 0.0 && point.offset.y() > 0.0, "independent u/v motion");
  expect((point.offset - repeated.offset).norm() == 0.0 && state.offset.norm() == 0.0,
    "repeated query is deterministic and leaves state untouched");
  expect((point.offset.array().abs() <= point.limits.array()).all(), "offset stays within safe region");
  expect((point.velocity.array().abs() <= config.max_speed).all(), "speed bound");
  expect((point.velocity.array().abs() <= config.max_acceleration * 0.1 + 1e-9).all(), "acceleration bound");
  const auto local = armor.rotation_world.transpose() * (point.position_world - armor.position_world);
  expect(std::abs(local.x()) < 1e-12 && (local.tail<2>() - point.offset).norm() < 1e-12,
    "offset lies in actual tilted plate plane");

  auto warmup = state;
  warmup.stable_frames = 0;
  expect(queryInPlateAim(prediction, armor, warmup, time, 0.02, config).offset.norm() == 0.0,
    "initialization uses center");
  auto disabled = config;
  disabled.enable = false;
  expect(queryInPlateAim(prediction, armor, state, time, 0.02, disabled).offset.norm() == 0.0,
    "disabled mode uses center");
  auto uncertain = prediction;
  uncertain.predicted_vehicle.covariance.setIdentity();
  expect(queryInPlateAim(uncertain, armor, state, time, 0.02, config).limits.norm() == 0.0,
    "uncertainty shrinks offset to center");
  auto side = armor;
  side.rotation_world = Eigen::AngleAxisd(std::numbers::pi / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  expect(queryInPlateAim(prediction, side, state, time, 0.02, config).limits.norm() < 1e-8,
    "side-on plate shrinks safe motion");

  state.direction = 1;
  auto slow = prediction;
  slow.predicted_vehicle.yaw_rate = -0.10;
  expect(queryInPlateAim(slow, armor, state, time, 0.02, config).direction == 0, "unconfirmed reversal stops old direction");
  slow.predicted_vehicle.yaw_rate = 0.02;
  expect(queryInPlateAim(slow, armor, state, time, 0.02, config).direction == 0, "near-zero dead zone");
  slow.predicted_vehicle.yaw_rate = -0.20;
  expect(queryInPlateAim(slow, armor, state, time, 0.02, config).direction == -1, "confirmed reversal");

  // 无重叠高度也不得越过当前板的安全边界。
  auto separate = prediction;
  for (auto& other : separate.armor_candidates) {
    if (other.armor_id != 0) other.position_world.z() += 0.3;
  }
  const auto tall = queryInPlateAim(separate, armor, state, time + 500ms, 0.02, config);
  expect(tall.offset.y() <= tall.limits.y() && tall.offset.y() > 0.0, "disjoint heights approach edge without leaving plate");
  config.mode = InPlateAimMode::FixedDiagonal;
  const auto diagonal = queryInPlateAim(prediction, armor, state, time + 800ms, 0.02, config);
  expect(diagonal.offset.x() > 0.0 && diagonal.offset.y() > 0.0, "fixed diagonal comparison mode");
  expect(std::abs(diagonal.offset.x() / diagonal.limits.x()
    - diagonal.offset.y() / diagonal.limits.y()) < 1e-12, "fixed mode stays on diagonal");
  auto tilted = armor;
  tilted.rotation_world = Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitX()) * armor.rotation_world;
  const auto tilt_aim = queryInPlateAim(prediction, tilted, state, time, 0.02, config);
  const Eigen::Vector3d tilt_local = tilted.rotation_world.transpose() * (tilt_aim.position_world - tilted.position_world);
  expect(std::abs(tilt_local.x()) < 1e-12 && (tilt_local.tail<2>() - tilt_aim.offset).norm() < 1e-12,
    "rolled vehicle keeps u/v in true plane");
  auto switch_state = state;
  switch_state.armor_id = 3;
  switch_state.offset.y() = -0.02;
  const auto& previous_plate = prediction.armor_candidates[3];
  const double previous_height = previous_plate.position_world.z()
    + previous_plate.rotation_world(2, 2) * switch_state.offset.y();
  config.mode = InPlateAimMode::Independent;
  const auto switched = queryInPlateAim(prediction, armor, switch_state, state.timestamp, 0.02, config);
  expect(std::abs(switched.position_world.z() - previous_height) < 1e-12,
    "plate handoff preserves shared world height");
  config.mode = InPlateAimMode::Center;
  expect(queryInPlateAim(prediction, armor, state, time, 0.02, config).offset.norm() == 0.0, "center comparison mode");
}

void interceptAndMpcChecks()
{
  auto target = numericalTarget();
  L1Sensor::RobotState robot;
  robot.timestamp = target.timestamp + 10ms;
  robot.bullet_speed = 25.0;
  robot.rpy.yaw = 0.02;
  PlannerContext context;
  context.config.enable_mpc = false;
  context.config.in_plate_aim.enable = true;
  InPlateAimState state;
  state.robot_id = target.robot_id;
  state.armor_id = 0;
  state.timestamp = target.timestamp;
  state.stable_frames = 10;
  const auto fire_time = target.timestamp + 100ms;
  const auto candidates = generateArmorCandidates(target, robot, context, fire_time, state);
  expect(candidates && candidates->size() == 4, "four candidates");
  if (!candidates || candidates->empty()) return;
  const auto& candidate = candidates->front();
  expect(candidate.valid && candidate.converged, "dynamic aim intercept converges");
  expect(std::abs(std::chrono::duration<double>(candidate.impact_time - fire_time).count()
    - candidate.ballistic.fly_time) < 0.0002, "aim, pose and flight time share converged instant");
  const auto predicted = Predictor{}.predict({target, candidate.impact_time});
  const auto expected = queryInPlateAim(predicted, predicted.armor_candidates.front(), state,
    fire_time, robot.rpy.yaw, context.config.in_plate_aim);
  expect((candidate.aim.position_world - expected.position_world).norm() < 1e-10, "final aim recomputed at final impact");
  expect((candidate.armor.position_world - predicted.armor_candidates.front().position_world).norm() < 1e-10,
    "real plate center preserved");
  const double center_yaw = std::atan2(candidate.armor.position_world.y(), candidate.armor.position_world.x());
  expect(std::abs(candidate.ballistic.yaw - robot.rpy.yaw) < std::abs(center_yaw - robot.rpy.yaw),
    "horizontal offset reduces immediate muzzle chasing");

  AimPlan direct;
  direct.valid = true;
  direct.armor_id = 0;
  direct.armor_geometry_valid = true;
  direct.fire_time = fire_time;
  direct.impact_time = candidate.impact_time;
  direct.fly_time = candidate.ballistic.fly_time;
  direct.yaw = candidate.ballistic.yaw;
  direct.pitch = candidate.ballistic.pitch;
  direct.aim_query_state = state;
  const auto reference = buildTinyMpcReference(direct, target, robot, Eigen::Isometry3d::Identity(), context.config);
  expect(reference.valid, "MPC sample intercepts converge");
  expect(std::abs(reference.yaw.angle[kTinyMpcControlIndex]) < 1e-12
    && std::abs(reference.pitch.angle[kTinyMpcControlIndex] - direct.pitch) < 1e-12,
    "MPC center matches direct dynamic aim exactly");

  auto fast_target = target;
  fast_target.yaw_rate = 6.0;
  const auto fast_candidates = generateArmorCandidates(fast_target, robot, context, fire_time, state, 0);
  if (fast_candidates && !fast_candidates->empty() && fast_candidates->front().valid) {
    auto fast_direct = direct;
    fast_direct.yaw = fast_candidates->front().ballistic.yaw;
    fast_direct.fly_time = fast_candidates->front().ballistic.fly_time;
    fast_direct.impact_time = fast_candidates->front().impact_time;
    const auto future = buildTinyMpcReference(fast_direct, fast_target, robot,
      Eigen::Isometry3d::Identity(), context.config);
    expect(future.valid && std::any_of(future.armor_id.begin() + kTinyMpcControlIndex + 1,
      future.armor_id.end(), [](int id) { return id != 0; }), "MPC anticipates future plate handoff");
  } else {
    expect(false, "fast target intercept valid");
  }

  Planner planner(context.config);
  auto first = planner.plan(target, robot, context);
  expect(first.valid && first.in_plate_aim.offset.norm() == 0.0, "planner starts at center");
  const auto stale = planner.plan(target, robot, context);
  expect(!stale.fire_permitted, "stale observation cannot fire");
  target = Predictor{}.predict(target, 0.01);
  robot.timestamp += 10ms;
  const auto fresh = planner.plan(target, robot, context);
  expect(fresh.aim_query_state.stable_frames == 1, "stale frame and candidate iterations never increment state");
  auto another_target = target;
  another_target.robot_id = 4;
  another_target.timestamp += 10ms;
  robot.timestamp += 10ms;
  const auto changed = planner.plan(another_target, robot, context);
  expect(changed.valid && changed.in_plate_aim.offset.norm() == 0.0
    && changed.aim_query_state.robot_id < 0, "target identity change resets aim state");
  planner.resetTracking();
  expect(planner.trackingState().robot_id < 0, "reset clears tracking");
}

L5Control::FireInput makeFireInput(double rotation_rate, double drag)
{
  L5Control::FireInput input;
  input.target = L3Estimation::EskfTarget(L3Estimation::ArmorName::Infantry3, 4.0,
    rotation_rate, 0.26, 0.0, 0.04);
  input.target->predict(TimePoint{1s});
  input.track_state = L3Estimation::TrackState::Tracking;
  const auto target = *runtime::toL4TargetState(input.target);
  L1Sensor::RobotState robot;
  robot.timestamp = target.timestamp;
  robot.bullet_speed = 25.0;
  PlannerConfig config;
  config.enable_mpc = false;
  config.enable_air_resistance = drag > 0.0;
  config.linear_drag_coefficient = drag;
  PlannerContext context;
  context.config = config;
  // 非零枪口平移验证世界系与枪口系的相对位置。
  context.T_barrel_world.translation() = Eigen::Vector3d{0.02, -0.01, 0.03};
  Planner planner(config);
  input.plan = planner.plan(target, robot, context);
  input.plan.fire_permitted = true; // 将选板稳定门控与几何检查独立测试
  input.actual_yaw = input.plan.yaw;
  input.actual_pitch = input.plan.pitch;
  return input;
}

void fireChecks()
{
  L5Control::FireConfig config;
  config.shoot_enable = true;
  L5Control::FireDecider decider(config);
  for (const double drag : {0.0, 0.5}) {
    auto input = makeFireInput(0.0, drag);
    expect(input.plan.valid, "synthetic fire plan valid");
    auto center = decider.decide(input);
    expect(center.shoot && center.predicted_hit_uv.norm() < 1e-4, "actual center hit accepted in vacuum and drag models");

    // 偏心瞄准以当前指令为误差零点，板心交点只作诊断。
    const auto solve_at_u = [&](double u) {
      BallisticRequest request;
      request.target_position_barrel = input.plan.armor_center_barrel
        + input.plan.armor_rotation_world.col(1) * u;
      request.bullet_speed = input.plan.bullet_speed;
      request.enable_air_resistance = drag > 0.0;
      request.linear_drag_coefficient = drag;
      return BallisticSolver{}.solve(request);
    };
    const double limit = input.plan.hit_half_width;
    auto inside = solve_at_u(limit - 0.003);
    input.plan.in_plate_aim.offset.x() = limit - 0.003;
    input.actual_yaw = inside.yaw;
    input.actual_pitch = inside.pitch;
    input.plan.yaw = inside.yaw;
    input.plan.pitch = inside.pitch;
    auto result = decider.decide(input);
    expect(result.shoot && result.predicted_hit_uv.x() > 0.0, "off-center actual hit accepted");
    auto outside = solve_at_u(limit + 0.005);
    input.actual_yaw = outside.yaw;
    input.actual_pitch = outside.pitch;
    input.plan.yaw = outside.yaw;
    input.plan.pitch = outside.pitch;
    result = decider.decide(input);
    expect(result.shoot && result.aim_error_within_tolerance
        && !result.hit_inside_plate && result.yaw_error == 0.0,
      "exact aim following is accepted independently of plate-center hit diagnostic");
    auto opposite = solve_at_u(-limit + 0.003);
    input.actual_yaw = opposite.yaw;
    input.actual_pitch = opposite.pitch;
    result = decider.decide(input);
    expect(!result.shoot && result.hit_inside_plate && !result.aim_error_within_tolerance,
      "physical plate hit cannot override excessive error relative to offset aim point");

    input.plan.using_MPC = true;
    input.plan.samples = {AimSample{}};
    input.plan.samples.front().yaw = outside.yaw;
    input.plan.samples.front().pitch = outside.pitch;
    input.plan.yaw = outside.yaw + 2.0 * config.min_yaw_tolerance;
    input.actual_yaw = outside.yaw;
    input.actual_pitch = outside.pitch;
    result = decider.decide(input);
    expect(result.shoot && result.yaw_error == 0.0,
      "MPC fire error follows current executable sample instead of direct reference or plate center");

    const double yaw_tolerance = result.tolerance.yaw;
    const double pitch_tolerance = result.tolerance.pitch;
    input.actual_yaw = outside.yaw + 0.99 * yaw_tolerance;
    input.actual_pitch = outside.pitch + 0.99 * pitch_tolerance;
    expect(decider.decide(input).shoot, "both aim errors just inside tolerance permit fire");
    input.actual_yaw = outside.yaw + 1.01 * yaw_tolerance;
    expect(!decider.decide(input).shoot, "yaw error above aim tolerance rejects fire");
    input.actual_yaw = outside.yaw;
    input.actual_pitch = outside.pitch + 1.01 * pitch_tolerance;
    expect(!decider.decide(input).shoot, "pitch error above aim tolerance rejects fire");

    input.plan.samples.front().yaw = std::numbers::pi - 0.001;
    input.actual_yaw = -std::numbers::pi + 0.001;
    input.actual_pitch = outside.pitch;
    result = decider.decide(input);
    expect(result.shoot && std::abs(result.yaw_error - 0.002) < 1e-12,
      "aim error wraps correctly across the yaw boundary");
  }
  auto moving = makeFireInput(3.0, 0.5);
  const auto decision = decider.decide(moving);
  expect(decision.shoot && decision.predicted_hit_uv.norm() < 0.005, "moving plate evaluated at actual ballistic intersection time");
  moving.actual_yaw = std::numeric_limits<double>::quiet_NaN();
  expect(!decider.decide(moving).shoot, "non-finite actual pose rejects fire");
}

void aimEdgeMarginChecks()
{
  L5Control::FireConfig fire_config;
  fire_config.shoot_enable = true;
  const L5Control::FireDecider decider(fire_config);
  InPlateAimConfig config;
  config.enable = true;
  config.mode = InPlateAimMode::Independent;
  for (const double drag : {0.0, 0.5}) {
    for (int axis = 0; axis < 2; ++axis) {
      for (const double sign : {-1.0, 1.0}) {
        auto input = makeFireInput(0.0, drag);
        auto target = *runtime::toL4TargetState(input.target);
        // 精确状态隔离几何边距，避免合成目标的初始化协方差把瞄点压回板心。
        target.filter_state.reset();
        target.covariance.setZero();
        const auto prediction = Predictor{}.predict({target, input.plan.impact_time});
        const auto& armor = prediction.armor_candidates[input.plan.armor_id];
        const Eigen::Vector3d translation =
          input.plan.armor_center_barrel - input.plan.armor_center_world;
        InPlateAimState state;
        state.robot_id = target.robot_id;
        state.armor_id = armor.armor_id;
        state.stable_frames = config.warmup_frames;
        state.timestamp = input.plan.fire_time;
        const auto center = queryInPlateAim(prediction, armor, state, state.timestamp,
          input.actual_yaw, config, -translation);
        state.offset[axis] = sign * center.limits[axis];
        // 停转瞬间仍保留上一帧偏心瞄点，不人为重置成板心。
        const auto aim = queryInPlateAim(prediction, armor, state, state.timestamp,
          input.actual_yaw, config, -translation);
        expect(aim.valid && std::abs(aim.offset[axis]) > 0.02,
          "edge regression exercises a real nonzero dynamic aim offset");

        const auto solve_at_offset = [&](const Eigen::Vector3d& point) {
          BallisticRequest request;
          request.target_position_barrel = point + translation;
          request.bullet_speed = input.plan.bullet_speed;
          request.enable_air_resistance = drag > 0.0;
          request.linear_drag_coefficient = drag;
          return BallisticSolver{}.solve(request);
        };
        const auto reference = solve_at_offset(aim.position_world);
        input.plan.in_plate_aim = aim;
        input.plan.aim_point_world = aim.position_world;
        input.plan.aim_point_barrel = aim.position_world + translation;
        input.plan.yaw = reference.yaw;
        input.plan.pitch = reference.pitch;
        const Eigen::Vector3d direction = sign * armor.rotation_world.col(axis + 1);
        const auto following = solve_at_offset(aim.position_world + 0.003 * direction);
        input.actual_yaw = following.yaw;
        input.actual_pitch = following.pitch;
        const auto hit = decider.decide(input);
        expect(hit.shoot && hit.hit_inside_plate,
          "dynamic aim edge margin accepts 3 mm following error inside physical hit region");

        const auto outside = solve_at_offset(aim.position_world + 0.015 * direction);
        input.actual_yaw = outside.yaw;
        input.actual_pitch = outside.pitch;
        const auto offset_following = decider.decide(input);
        expect(offset_following.shoot && offset_following.aim_error_within_tolerance
            && !offset_following.hit_inside_plate,
          "allowed aim following error is not rejected by plate-center hit diagnostic");
      }
    }
  }
}

void aimToleranceChecks()
{
  auto input = makeFireInput(0.0, 0.0);
  L5Control::FireConfig config;
  config.shoot_enable = true;
  config.min_yaw_tolerance = config.min_pitch_tolerance = 0.0;
  const L5Control::FireDecider decider(config);
  const auto center = decider.decide(input);
  // 固定瞄点距离和朝向，只改变相对板心的偏移元数据，容差不应被扣减。
  input.plan.in_plate_aim.offset = {0.03, -0.03};
  const auto offset = decider.decide(input);
  expect(offset.shoot && offset.tolerance.yaw == center.tolerance.yaw
      && offset.tolerance.pitch == center.tolerance.pitch,
    "symmetric aim tolerance does not shrink with plate-center offset");

  config.min_yaw_tolerance = 2.0 * std::numbers::pi / 180.0;
  config.min_pitch_tolerance = 1.8 * std::numbers::pi / 180.0;
  const auto with_floor = L5Control::FireDecider(config).decide(input);
  expect(with_floor.tolerance.yaw == config.min_yaw_tolerance
      && with_floor.tolerance.pitch == config.min_pitch_tolerance,
    "configured yaw and pitch tolerance floors participate in fire decision");
}

void optionalFireGateChecks()
{
  L5Control::FireConfig strict;
  strict.shoot_enable = true;
  auto relaxed = strict;
  relaxed.stop_fire_on_temp_lost = false;
  relaxed.stop_fire_on_aim_error = false;
  const L5Control::FireDecider decider(relaxed);
  auto input = makeFireInput(0.0, 0.0);
  input.actual_yaw += 0.1;
  input.actual_pitch += 0.1;
  expect(!L5Control::FireDecider(strict).decide(input).shoot,
    "enabled aim gate rejects excessive following error");
  auto decision = decider.decide(input);
  expect(decision.shoot && !decision.aim_error_within_tolerance
      && decision.reasons.empty(),
    "disabled aim gate keeps errors as diagnostics without stopping fire");

  input.track_state = L3Estimation::TrackState::TempLost;
  input.plan.tracked_phase = ArmorTrackingPhase::Tracking;
  input.plan.tracked_ready = false;
  input.plan.within_firing_window = true;
  input.plan.fire_permitted = false;
  expect(decider.decide(input).shoot,
    "TempLost valid locked prediction continues fire despite missing fresh observation and aim error");
  auto temp_lost_only = relaxed;
  temp_lost_only.stop_fire_on_aim_error = true;
  expect(!L5Control::FireDecider(temp_lost_only).decide(input).shoot,
    "allowing TempLost does not bypass an enabled aim-error gate");
  auto aim_only = relaxed;
  aim_only.stop_fire_on_temp_lost = true;
  expect(!L5Control::FireDecider(aim_only).decide(input).shoot,
    "disabling aim-error gate does not bypass an enabled TempLost gate");

  const auto expect_blocked = [&](auto change, std::string_view message) {
    auto changed = input;
    change(changed);
    expect(!decider.decide(changed).shoot, message);
  };
  expect_blocked([](auto& x) { x.plan.within_firing_window = false; },
    "TempLost continuation does not bypass firing window");
  expect_blocked([](auto& x) { x.plan.tracked_phase = ArmorTrackingPhase::Switching; },
    "TempLost continuation does not bypass switching lock gate");
  expect_blocked([](auto& x) { x.plan.valid = false; },
    "invalid plan still stops fire with optional gates disabled");
  expect_blocked([](auto& x) { x.track_state = L3Estimation::TrackState::Lost; },
    "fully lost tracking still stops fire");
  expect_blocked([](auto& x) { x.track_state = L3Estimation::TrackState::Detecting; },
    "unconfirmed target still stops fire");
  expect_blocked([](auto& x) { x.target.reset(); }, "missing target still stops fire");
  expect_blocked([](auto& x) { x.command_jump = true; }, "command jump still stops fire");
  expect_blocked([](auto& x) { x.bullet_speed_valid = false; }, "invalid bullet speed still stops fire");
  expect_blocked([](auto& x) { x.actual_pitch = std::numeric_limits<double>::quiet_NaN(); },
    "nonfinite pose still stops fire");
  relaxed.shoot_enable = false;
  decision = L5Control::FireDecider(relaxed).decide(input);
  expect(!decision.shoot && decision.fire_feasible,
    "master shoot switch remains independent of optional gates");
}
}
int main()
{
  policyChecks();
  interceptAndMpcChecks();
  fireChecks();
  aimEdgeMarginChecks();
  aimToleranceChecks();
  optionalFireGateChecks();
  const auto tuning = loadPlannerTuning("config/planner_config.yaml");
  expect(tuning.planner.in_plate_aim.enable, "configuration loads dynamic aim");
  if (failures == 0) std::cout << "in_plate_aim_smoke: all checks passed\n";
  return failures == 0 ? 0 : 1;
}
