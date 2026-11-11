#include "l3_planning/armor_candidate_generator.hpp"
#include "l3_planning/in_plate_aim.hpp"
#include "l3_planning/planner.hpp"
#include "l3_planning/planner_config.hpp"
#include "l3_planning/tiny_mpc.hpp"
#include "l4_control/fire_decision.hpp"
#include "runtime/l4_target_adapter.hpp"

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
void expect(bool ok, std::string_view message)
{
  if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}

L3Estimation::TargetState outpost(double rate)
{
  L3Estimation::TargetState target;
  target.robot_id = static_cast<int>(L3Estimation::ArmorName::Outpost);
  target.armor_count = 3;
  target.center = {4.0, 0.0, 0.20};
  target.radius = 0.2765;
  target.yaw_rate = rate;
  target.three_armor_height_offsets = {0.0, 0.01, 0.04};
  target.timestamp = TimePoint{2s};
  target.covariance.setZero();
  return target;
}

InPlateAimState settledState(const L3Estimation::TargetState& target, int id)
{
  InPlateAimState state;
  state.robot_id = target.robot_id;
  state.armor_id = id;
  state.timestamp = target.timestamp - 1s;
  state.stable_frames = 10;
  return state;
}

void geometryAndHeightChecks()
{
  InPlateAimConfig config;
  config.enable = true;
  const double step = 2.0 * std::numbers::pi / 3.0;
  for (const double rate : {2.51, -2.51}) {
    double previous_height = 0.0;
    for (int id = 0; id < 3; ++id) {
      auto target = outpost(rate);
      target.yaw = -id * step;
      const auto prediction = Predictor{}.predict({target, target.timestamp});
      expect(prediction.valid && prediction.armor_candidates.size() == 3, "outpost expands exactly three armors");
      const auto& armor = prediction.armor_candidates[id];
      expect(std::abs(armor.position_world.z() - target.center.z()
        - target.three_armor_height_offsets[id]) < 1e-12, "each plate retains its independent height");
      expect(armor.rotation_world(2, 0) > 0.0 && armor.rotation_world(0, 2) < 0.0,
        "outpost uses forward tilt instead of normal vehicle backward tilt");
      const auto state = settledState(target, id);
      const auto aim = queryInPlateAim(prediction, armor, state, target.timestamp, 0.01, config);
      const auto repeated = queryInPlateAim(prediction, armor, state, target.timestamp, 0.01, config);
      const int expected_next = (id + (rate > 0.0 ? 2 : 1)) % 3;
      expect(aim.valid && aim.next_armor_id == expected_next, "120-degree cyclic successor follows rotation direction");
      expect(aim.using_common_height, "three plate shared height recognized even when other plates face away");
      expect((aim.offset.array().abs() <= aim.limits.array() + 1e-12).all(), "three-plate aim remains within safe bounds");
      expect((repeated.position_world - aim.position_world).norm() == 0.0 && state.offset.norm() == 0.0,
        "outpost iteration queries do not advance state");
      expect(std::abs(aim.position_world.z() - 0.22) < 1e-4, "common height near middle of all three intervals");
      if (id > 0) expect(std::abs(aim.position_world.z() - previous_height) < 1e-5,
        "shared world height stays stable through a complete three-plate cycle");
      previous_height = aim.position_world.z();

      auto pair_config = config;
      pair_config.outpost_common_height = false;
      const auto pair = queryInPlateAim(prediction, armor, state, target.timestamp, 0.01, pair_config);
      expect(!pair.using_common_height, "pairwise comparison switch disables three-plate common height");
      if (id == 1 && rate > 0.0) expect(std::abs(pair.position_world.z() - aim.position_world.z()) > 0.008,
        "common height avoids pairwise cyclic height changes");

      auto warmup = state;
      warmup.stable_frames = 0;
      expect(queryInPlateAim(prediction, armor, warmup, target.timestamp, 0.01, config).offset.norm() == 0.0,
        "outpost initialization remains at plate center");
      auto uncertain = prediction;
      uncertain.predicted_vehicle.covariance.setIdentity();
      expect(queryInPlateAim(uncertain, armor, state, target.timestamp, 0.01, config).offset.norm() == 0.0,
        "uncertain outpost returns to center");
    }
  }

  auto target = outpost(2.51);
  target.three_armor_height_offsets = {0.0, 0.04, 0.15};
  target.yaw = -step; // 当前板 1，下一板 0 有重叠，但三板无共有高度。
  auto prediction = Predictor{}.predict({target, target.timestamp});
  auto aim = queryInPlateAim(prediction, prediction.armor_candidates[1], settledState(target, 1),
    target.timestamp, 0.0, config);
  expect(!aim.using_common_height && aim.offset.y() < 0.0, "no three-plate overlap falls back to next plate overlap");
  target.yaw = 0.0;
  target.three_armor_height_offsets = {0.0, 0.12, -0.12};
  prediction = Predictor{}.predict({target, target.timestamp});
  aim = queryInPlateAim(prediction, prediction.armor_candidates[0], settledState(target, 0),
    target.timestamp, 0.0, config);
  expect(!aim.using_common_height && aim.offset.y() < 0.0
    && std::abs(aim.offset.y()) <= aim.limits.y(), "separate heights approach next board without leaving current board");
  target.yaw_rate = 0.0;
  prediction = Predictor{}.predict({target, target.timestamp});
  aim = queryInPlateAim(prediction, prediction.armor_candidates[0], settledState(target, 0),
    target.timestamp, 0.0, config);
  expect(aim.direction == 0 && aim.next_armor_id == -1 && aim.offset.norm() == 0.0, "stopped outpost does not cycle successors");
  expect(nextArmorInRotation(3, 3, 1) == -1, "invalid three-plate ID rejected");
}

void covarianceChecks()
{
  // 前哨 roll/pitch 为不可观测冗余量，不能像四板倾斜车那样投影其方差。
  L3Estimation::EskfTarget filter(L3Estimation::ArmorName::Outpost, 4.0, 2.51, 0.2765, -2.51);
  filter.predict(TimePoint{1s});
  const auto target = *runtime::toL4TargetState(filter);
  const auto prediction = Predictor{}.predict({target, target.timestamp});
  InPlateAimConfig config;
  config.enable = true;
  config.uncertainty_sigma = 0.001;
  namespace VM = L3Estimation::VehicleModel;
  for (int id = 0; id < 3; ++id) {
    const auto& armor = prediction.armor_candidates[id];
    const auto aim = queryInPlateAim(prediction, armor, settledState(target, id), target.timestamp, 0.0, config);
    Eigen::Matrix<double, 3, VM::kStateSize> jacobian;
    constexpr double epsilon = 1e-6;
    const auto nominal = prediction.predicted_vehicle.filter_state->rawState();
    for (int column = 0; column < VM::kStateSize; ++column) {
      auto plus = nominal;
      auto minus = nominal;
      L3Estimation::EskfTarget::State delta = L3Estimation::EskfTarget::State::Zero();
      delta[column] = epsilon;
      VM::injectState(delta, plus);
      delta[column] = -epsilon;
      VM::injectState(delta, minus);
      const Eigen::Vector3d right = VM::armorPose<double>(plus.data(), id, 3, filter.name).translation();
      const Eigen::Vector3d left = VM::armorPose<double>(minus.data(), id, 3, filter.name).translation();
      jacobian.col(column) = (right - left) / (2.0 * epsilon);
    }
    const Eigen::Matrix3d covariance = jacobian * filter.covariance() * jacobian.transpose();
    const double facing = std::clamp(armor.rotation_world.col(0).dot(armor.position_world.normalized()), 0.0, 1.0);
    for (int axis = 0; axis < 2; ++axis) {
      const Eigen::Vector3d direction = armor.rotation_world.col(axis + 1);
      const double half_size = 0.5 * (axis == 0 ? config.small_width : config.height) * config.hit_margin_ratio;
      const double expected = std::max(0.0, half_size - config.edge_margin
        - config.uncertainty_sigma * std::sqrt(direction.dot(covariance * direction))) * facing;
      expect(std::abs(aim.limits[axis] - expected) < 1e-9,
        "outpost uncertainty matches numerical differentiation of its yaw-only three-height model");
    }
  }
}

void fixedWindowChecks()
{
  bool exercised_fast_window_edge = false;
  for (const bool is_outpost : {false, true}) {
    PlannerContext context;
    const double enter = (is_outpost ? context.config.outpost_enter_angle
      : context.config.normal_enter_angle) * std::numbers::pi / 180.0;
    const double leave = (is_outpost ? context.config.outpost_leave_angle
      : context.config.normal_leave_angle) * std::numbers::pi / 180.0;
    for (const double rate : {0.0, 0.02, -0.02, 2.51, -2.51, 6.0, -6.0}) {
      for (const double yaw : {-1.0, 0.0, 1.0}) {
        auto target = outpost(rate);
        target.yaw = yaw;
        if (!is_outpost) {
          target.robot_id = 1;
          target.armor_count = 4;
        }
        L1Sensor::RobotState robot;
        robot.timestamp = target.timestamp;
        robot.bullet_speed = 25.0;
        const auto candidates = generateArmorCandidates(
          target, robot, context, target.timestamp);
        expect(candidates && candidates->size() == target.armor_count,
          "fixed-window checks generate all vehicle plates");
        if (!candidates) continue;
        for (const auto& candidate : *candidates) {
          const bool rotating = std::abs(rate) > context.config.rotation_rate_dead_zone;
          const double phase = rotating
            ? (rate > 0.0 ? candidate.delta_angle : -candidate.delta_angle)
            : std::abs(candidate.delta_angle);
          const bool inside = rotating
            ? phase >= -enter && phase <= leave : phase <= enter;
          expect(candidate.within_selection_window == inside,
            "selection uses fixed angles for both vehicle types and rotation directions");
          expect(candidate.within_firing_window == inside,
            "firing uses the same fixed window at every rotation speed");
          if (std::abs(rate) >= 6.0 && inside
              && (phase < -enter * 0.65 || phase > leave * 0.65)) {
            exercised_fast_window_edge = true;
          }
        }
      }
    }
  }
  expect(exercised_fast_window_edge,
    "fast plates remain selectable in the region previously removed by speed scaling");
}

void interceptionAndReferenceChecks()
{
  for (const double rate : {2.51, -2.51, 6.0, -6.0}) {
    const auto target = outpost(rate);
    PlannerContext context;
    context.config.in_plate_aim.enable = true;
    context.config.enable_air_resistance = true;
    context.config.linear_drag_coefficient = 0.5;
    L1Sensor::RobotState robot;
    robot.timestamp = target.timestamp;
    robot.bullet_speed = 25.0;
    const auto fire_time = target.timestamp + 15ms;
    const auto state = settledState(target, 0);
    const auto candidates = generateArmorCandidates(target, robot, context, fire_time, state);
    expect(candidates && candidates->size() == 3, "outpost has three ballistic candidates");
    if (!candidates || candidates->size() != 3) continue;
    const auto& candidate = candidates->front();
    expect(candidate.valid && candidate.converged, "outpost dynamic intercept converges in both rotation directions");
    expect(std::abs(std::chrono::duration<double>(candidate.impact_time - fire_time).count()
      - candidate.ballistic.fly_time) <= 0.0002, "outpost aim and pose use converged impact time");
    const auto prediction = Predictor{}.predict({target, candidate.impact_time});
    const auto aim = queryInPlateAim(prediction, prediction.armor_candidates.front(), state,
      fire_time, robot.rpy.yaw, context.config.in_plate_aim);
    expect((aim.position_world - candidate.aim.position_world).norm() < 1e-12,
      "outpost final aim is recomputed at final impact instant");

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
    expect(reference.valid, "three-plate MPC reference valid");
    expect(std::abs(reference.yaw.angle[kTinyMpcControlIndex]) < 1e-12
      && std::abs(reference.pitch.angle[kTinyMpcControlIndex] - direct.pitch) < 1e-12,
      "outpost MPC center agrees with direct aim");
    int previous_id = 0;
    int switches = 0;
    for (std::size_t index = kTinyMpcControlIndex + 1; index < kTinyMpcHorizon; ++index) {
      const int id = reference.armor_id[index];
      expect(id >= 0 && id < 3, "MPC never selects phantom fourth plate");
      if (id != previous_id) {
        expect(id == nextArmorInRotation(previous_id, 3, rate > 0.0 ? 1 : -1),
          "MPC successor order follows outpost rotation");
        ++switches;
        previous_id = id;
      }
    }
    expect(switches >= 1 && switches <= 3, "outpost MPC anticipates handoff without repeatedly switching to incoming boards");
  }
}

void actualHitChecks()
{
  L5Control::FireConfig fire_config;
  fire_config.shoot_enable = true;
  const L5Control::FireDecider decider(fire_config);
  for (const double rate : {2.51, -2.51}) {
    L5Control::FireInput input;
    input.target = L3Estimation::EskfTarget(L3Estimation::ArmorName::Outpost, 4.0, rate, 0.2765);
    input.target->predict(TimePoint{1s});
    input.track_state = L3Estimation::TrackState::Tracking;
    const auto target = *runtime::toL4TargetState(input.target);
    PlannerConfig config;
    config.enable_mpc = false;
    config.in_plate_aim.enable = true;
    config.enable_air_resistance = true;
    config.linear_drag_coefficient = 0.5;
    L1Sensor::RobotState robot;
    robot.timestamp = target.timestamp;
    robot.bullet_speed = 25.0;
    input.plan = Planner{config}.plan(target, robot);
    input.plan.fire_permitted = true; // 将跟随误差与选板窗口门控分开验证。
    input.actual_yaw = input.plan.yaw;
    input.actual_pitch = input.plan.pitch;
    expect(input.plan.valid && input.plan.armor_id >= 0 && input.plan.armor_id < 3, "ESKF outpost planner selects a real plate");
    const auto hit = decider.decide(input);
    expect(hit.shoot && hit.predicted_hit_uv.norm() < 0.005, "actual ballistic hit uses forward-tilted outpost plane");

    BallisticRequest request;
    request.target_position_barrel = input.plan.armor_center_barrel
      + input.plan.armor_rotation_world.col(2) * (input.plan.hit_half_height + 0.03);
    request.bullet_speed = input.plan.bullet_speed;
    request.enable_air_resistance = true;
    request.linear_drag_coefficient = 0.5;
    const auto outside = BallisticSolver{}.solve(request);
    input.actual_yaw = input.plan.yaw = outside.yaw;
    input.actual_pitch = input.plan.pitch = outside.pitch;
    const auto following = decider.decide(input);
    expect(following.shoot && following.aim_error_within_tolerance
        && !following.hit_inside_plate,
      "outpost fire follows current aim instead of plate-center intersection");
    input.actual_pitch += 2.0 * following.tolerance.pitch;
    expect(!decider.decide(input).shoot, "outpost pitch error above aim tolerance rejects fire");
    input.actual_pitch = input.plan.pitch;
    input.plan.armor_id = 3;
    expect(!decider.decide(input).shoot, "nonexistent fourth plate cannot pass fire check");
    input.plan.armor_id = 0;
    input.plan.fire_permitted = false;
    expect(!decider.decide(input).shoot, "outpost handoff gap retains firing gate");
  }
}
}

int main()
{
  geometryAndHeightChecks();
  covarianceChecks();
  fixedWindowChecks();
  interceptionAndReferenceChecks();
  actualHitChecks();
  const auto tuning = loadPlannerTuning("config/planner_config.yaml");
  expect(tuning.planner.in_plate_aim.outpost_common_height, "outpost common-height option loads");
  if (failures == 0) std::cout << "outpost_aim_smoke: all checks passed\n";
  return failures == 0 ? 0 : 1;
}
