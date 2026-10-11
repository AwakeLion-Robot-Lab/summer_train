#include "l4_planning/planner_validation.hpp"

#include "l1_sensor/serial/robot_state.hpp"
#include "l4_planning/planner.hpp"

#include <cmath>

namespace L4Planning {

bool validPlannerConfig(const PlannerConfig& config) noexcept
{
  const bool dynamic_windows_valid =
    !config.enable_dynamic_windows
    || (std::isfinite(config.window_shrink_start_speed)
        && config.window_shrink_start_speed >= 0.0
        && std::isfinite(config.window_shrink_end_speed)
        && config.window_shrink_end_speed
             > config.window_shrink_start_speed
        && std::isfinite(config.selection_min_window_scale)
        && config.selection_min_window_scale > 0.0
        && config.selection_min_window_scale <= 1.0
        && std::isfinite(config.firing_max_window_scale)
        && config.firing_max_window_scale > 0.0
        && config.firing_max_window_scale < 1.0
        && std::isfinite(config.firing_min_window_scale)
        && config.firing_min_window_scale > 0.0
        && config.firing_min_window_scale
             <= config.firing_max_window_scale
        && config.firing_min_window_scale
             < config.selection_min_window_scale
        && std::isfinite(config.selection_hold_margin)
        && config.selection_hold_margin >= 0.0
        && config.normal_enter_angle + config.selection_hold_margin < 180.0
        && config.outpost_enter_angle + config.selection_hold_margin < 180.0);
  const bool mpc_valid =
    !config.enable_mpc
    || (std::isfinite(config.yaw_angle_weight)
        && config.yaw_angle_weight >= 0.0
        && std::isfinite(config.yaw_velocity_weight)
        && config.yaw_velocity_weight >= 0.0
        && std::isfinite(config.yaw_acceleration_weight)
        && config.yaw_acceleration_weight > 0.0
        && std::isfinite(config.pitch_angle_weight)
        && config.pitch_angle_weight >= 0.0
        && std::isfinite(config.pitch_velocity_weight)
        && config.pitch_velocity_weight >= 0.0
        && std::isfinite(config.pitch_acceleration_weight)
        && config.pitch_acceleration_weight > 0.0
        && std::isfinite(config.min_yaw_acceleration)
        && std::isfinite(config.max_yaw_acceleration)
        && config.min_yaw_acceleration <= config.max_yaw_acceleration
        && std::isfinite(config.min_pitch_acceleration)
        && std::isfinite(config.max_pitch_acceleration)
        && config.min_pitch_acceleration <= config.max_pitch_acceleration
        && config.mpc_max_iterations > 0
        && std::isfinite(config.mpc_admm_rho)
        && config.mpc_admm_rho > 0.0
        && std::isfinite(config.mpc_primal_tolerance)
        && config.mpc_primal_tolerance > 0.0
        && std::isfinite(config.mpc_dual_tolerance)
        && config.mpc_dual_tolerance > 0.0);
  const bool ballistic_valid =
    std::isfinite(config.gravity)
    && config.gravity > 0.0
    && (!config.enable_air_resistance
        || (std::isfinite(config.linear_drag_coefficient)
            && config.linear_drag_coefficient > 0.0));

  return config.max_iterations > 0
         && config.fly_time_tolerance.count() > 0
         && std::isfinite(config.position_tolerance)
         && config.position_tolerance > 0.0
         && std::isfinite(config.angle_tolerance)
         && config.angle_tolerance >= 0.0
         && std::isfinite(config.switch_yaw_dead_zone)
         && config.switch_yaw_dead_zone >= 0.0
         && std::isfinite(config.switch_pitch_dead_zone)
         && config.switch_pitch_dead_zone >= 0.0
         && std::isfinite(config.score_switch_threshold)
         && config.score_switch_threshold >= 0.0
         && config.score_switch_stable_frames > 0
         && std::isfinite(config.rotation_rate_dead_zone)
         && config.rotation_rate_dead_zone >= 0.0
         && config.lock_stable_frames > 0
         && std::isfinite(config.aim_cost_good_angle)
         && config.aim_cost_good_angle >= 0.0
         && std::isfinite(config.aim_cost_bad_angle)
         && config.aim_cost_bad_angle > config.aim_cost_good_angle
         && std::isfinite(config.normal_enter_angle)
         && config.normal_enter_angle > 0.0
         && std::isfinite(config.normal_leave_angle)
         && config.normal_leave_angle >= 0.0
         && config.normal_leave_angle <= config.normal_enter_angle
         && std::isfinite(config.outpost_enter_angle)
         && config.outpost_enter_angle > 0.0
         && std::isfinite(config.outpost_leave_angle)
         && config.outpost_leave_angle >= 0.0
         && config.outpost_leave_angle <= config.outpost_enter_angle
         && dynamic_windows_valid
         && config.max_lost_frames >= 0
         && ballistic_valid
         && mpc_valid;
}

bool validArmorScoreWeights(const ArmorScoreWeights& weights) noexcept
{
  if (!std::isfinite(weights.facing_weight)
      || !std::isfinite(weights.window_weight)
      || !std::isfinite(weights.aim_cost_weight)
      || weights.facing_weight < 0.0
      || weights.window_weight < 0.0
      || weights.aim_cost_weight < 0.0) {
    return false;
  }

  const double sum =
    weights.facing_weight + weights.window_weight + weights.aim_cost_weight;
  return std::abs(sum - 1.0) <= 1e-9;
}

bool validFacingAngleThresholds(
  double good_angle_degree,
  double bad_angle_degree) noexcept
{
  return std::isfinite(good_angle_degree)
         && good_angle_degree >= 0.0
         && std::isfinite(bad_angle_degree)
         && bad_angle_degree > good_angle_degree;
}

bool validPlanningContext(const PlannerContext& context) noexcept
{
  return validPlannerConfig(context.config)
         && validArmorScoreWeights(context.armor_score_weights)
         && validFacingAngleThresholds(
           context.facing_angle_good,
           context.facing_angle_bad)
         && context.T_barrel_world.translation().allFinite();
}

bool validRobotStateForPlanning(
  const L1Sensor::RobotState& robot_state) noexcept
{
  return std::isfinite(robot_state.bullet_speed)
         && robot_state.bullet_speed > 0.0
         && std::isfinite(robot_state.rpy.yaw)
         && std::isfinite(robot_state.rpy.pitch);
}

}  // namespace L4Planning
