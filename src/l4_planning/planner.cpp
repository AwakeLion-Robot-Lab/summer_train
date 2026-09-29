#include "l4_planning/planner.hpp"

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/target_state.hpp"
#include "l4_planning/armor_candidate_generator.hpp"
#include "l4_planning/latency_compensator.hpp"
#include "l4_planning/planner_validation.hpp"
#include "l4_planning/tiny_mpc.hpp"

#include <chrono>
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
    fire_time);
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
  plan.aim_point_world = selected.armor.position_world;
  plan.aim_point_barrel =
    selected.armor.position_world + context.T_barrel_world.translation();
  plan.yaw = selected.ballistic.yaw;
  plan.pitch = selected.ballistic.pitch;
  plan.fly_time = selected.ballistic.fly_time;
  plan.tracked_ready = selection.tracked_ready;
  plan.within_firing_window = selected.within_firing_window;
  plan.fire_permitted = plan.tracked_ready && plan.within_firing_window;
  plan.valid = true;

  return applyTinyMpc(
    plan,
    target_state,
    robot_state,
    context.T_barrel_world,
    config);
}

}  // namespace L4Planning
