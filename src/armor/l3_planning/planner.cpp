#include "l3_planning/planner.hpp"

#include "sensor/serial/robot_state.hpp"
#include "l2_estimation/target_state.hpp"
#include "l3_planning/armor_candidate_generator.hpp"
#include "l3_planning/latency_compensator.hpp"
#include "l3_planning/planner_validation.hpp"
#include "l3_planning/tiny_mpc.hpp"

#include <chrono>
#include <cmath>
#include <numbers>
#include <optional>
#include <utility>

namespace L4Planning {
namespace {

[[nodiscard]] TimePoint addSeconds(TimePoint time, double seconds)
{
  return time + std::chrono::duration_cast<TimePoint::duration>(
                  std::chrono::duration<double>(seconds));
}

}  // namespace

Planner::Planner(PlannerConfig config)
  : config_(std::move(config))
{
}

void Planner::setConfig(PlannerConfig config)
{
  config_ = std::move(config);
  resetTracking();
}

const PlannerConfig& Planner::config() const noexcept
{
  return config_;
}

void Planner::resetTracking() noexcept
{
  tracking_state_ = {};
  aim_state_ = {};
  last_target_.reset();
  last_observation_timestamp_.reset();
  target_lost_frames_ = 0;
}

const ArmorTrackingState& Planner::trackingState() const noexcept
{
  return tracking_state_;
}

AimPlan Planner::plan(
  const std::optional<L3Estimation::TargetState>& target,
  const L1Sensor::RobotState& robot_state)
{
  PlannerContext context;
  context.planning_time = robot_state.timestamp;
  context.config = config_;
  return plan(target, robot_state, context);
}

AimPlan Planner::plan(
  const std::optional<L3Estimation::TargetState>& target,
  const L1Sensor::RobotState& robot_state,
  const PlannerContext& context)
{
  const PlannerConfig& config = context.config;

  AimPlan plan;
  plan.generated_at = robot_state.timestamp;
  plan.using_MPC = false;

  if (!validRobotStateForPlanning(robot_state)
      || !validPlanningContext(context)) {
    return plan;
  }

  const bool observation_fresh =
    target.has_value()
    && (!last_observation_timestamp_.has_value()
        || target->timestamp > *last_observation_timestamp_);
  if (observation_fresh) {
    if (tracking_state_.robot_id >= 0
        && tracking_state_.robot_id != target->robot_id) {
      tracking_state_ = {};
      aim_state_ = {};
    }
    last_target_ = *target;
    last_observation_timestamp_ = target->timestamp;
    target_lost_frames_ = 0;
  } else {
    if (!last_target_.has_value()) {
      resetTracking();
      return plan;
    }
    ++target_lost_frames_;
    if (target_lost_frames_ >= config.max_lost_frames) {
      resetTracking();
      return plan;
    }
  }

  const L3Estimation::TargetState& target_state = *last_target_;
  tracking_state_.robot_id = target_state.robot_id;
  plan.target_id = target_state.robot_id;
  plan.tracked = true;
  plan.tracked_phase = tracking_state_.phase;

  const LatencyCompensator latency_compensator{context.latency};
  const LatencyResult latency = latency_compensator.calculate(
    target_state.timestamp,
    robot_state.timestamp,
    target_state.yaw_rate);
  if (!latency.valid) {
    return plan;
  }
  const TimePoint fire_time = addSeconds(
    latency.delay.camera_timestamp,
    latency.delay.total());

  auto candidates = generateArmorCandidates(
    target_state,
    robot_state,
    context,
    fire_time,
    aim_state_);
  if (!candidates.has_value()) {
    return plan;
  }

  SelectionRequest selection_request;
  selection_request.candidates = std::move(*candidates);
  selection_request.preferred_armor_id = tracking_state_.current_armor_id;
  selection_request.observation_fresh = observation_fresh;
  const TimePoint selection_time =
    context.planning_time != TimePoint{}
      ? context.planning_time
      : robot_state.timestamp;
  const SelectionResult selection =
    selectArmor(selection_request, selection_time, config);
  plan.tracked_phase = selection.phase;
  if (selection.phase == ArmorTrackingPhase::Unlocked) {
    plan.tracked = false;
    plan.target_id = -1;
  }
  if (!selection.valid || !selection.selected.has_value()) {
    return plan;
  }
  const ArmorCandidate& selected = *selection.selected;

  plan.armor_id = selected.armor.armor_id;
  plan.impact_time = selected.impact_time;
  plan.aim_point_world = selected.aim.position_world;
  plan.aim_point_barrel =
    selected.aim.position_world + context.T_barrel_world.translation();
  plan.yaw = selected.ballistic.yaw;
  plan.pitch = selected.ballistic.pitch;
  plan.fly_time = selected.ballistic.fly_time;
  plan.tracked_ready = selection.tracked_ready;
  plan.within_firing_window = selected.within_firing_window;
  plan.fire_permitted = plan.tracked_ready && plan.within_firing_window;
  plan.valid = true;
  plan.armor_center_world = selected.armor.position_world;
  plan.armor_center_barrel = selected.armor.position_world + context.T_barrel_world.translation();
  plan.armor_rotation_world = selected.armor.rotation_world;
  plan.armor_geometry_valid = true;
  plan.in_plate_aim = selected.aim;
  plan.aim_query_state = aim_state_;
  plan.fire_time = fire_time;
  plan.bullet_speed = robot_state.bullet_speed;
  plan.gravity = config.gravity;
  plan.drag_coefficient = config.enable_air_resistance ? config.linear_drag_coefficient : 0.0;
  const auto& aim_config = config.in_plate_aim;
  // edge_margin 只收缩瞄点运动范围，留下跟随误差余量；开火边界仍是有效受击面。
  // 若这里再次扣除边距，瞄点到达安全边缘后，几毫米跟随误差也会持续禁火。
  plan.hit_half_width = 0.5 * (selected.armor.armor_type == ArmorType::Large
    ? aim_config.large_width : aim_config.small_width) * aim_config.hit_margin_ratio;
  plan.hit_half_height = 0.5 * aim_config.height * aim_config.hit_margin_ratio;

  // 优化器读取旧快照；选定的 MPC 板内位置也必须在本帧成功后提交。
  plan = applyTinyMpc(plan, target_state, robot_state, context.T_barrel_world, config);

  // 候选迭代和 MPC 始终读取同一旧快照。新观测的已选结果只提交一次。
  if (observation_fresh && (aim_state_.robot_id != plan.target_id
      || fire_time > aim_state_.timestamp)) {
    const double dt = std::chrono::duration<double>(robot_state.timestamp - aim_state_.gun_timestamp).count();
    const double yaw_rate = aim_state_.robot_id == plan.target_id && dt > 1e-6
      ? std::remainder(robot_state.rpy.yaw - aim_state_.gun_yaw, 2.0 * std::numbers::pi) / dt : 0.0;
    aim_state_.stable_frames = aim_state_.robot_id == plan.target_id ? aim_state_.stable_frames + 1 : 1;
    aim_state_.robot_id = plan.target_id;
    aim_state_.armor_id = plan.armor_id;
    aim_state_.timestamp = fire_time;
    aim_state_.offset = plan.in_plate_aim.offset;
    aim_state_.velocity = plan.in_plate_aim.velocity;
    aim_state_.direction = plan.in_plate_aim.direction;
    aim_state_.position_world = plan.in_plate_aim.position_world;
    aim_state_.gun_timestamp = robot_state.timestamp;
    aim_state_.gun_yaw = robot_state.rpy.yaw;
    aim_state_.gun_yaw_rate = yaw_rate;
  }

  return plan;
}

}  // namespace L4Planning
