#include "l4_planning/planner_config.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace L4Planning {
namespace {

template <typename T>
void readOptional(
  const YAML::Node& node,
  const char* key,
  T& destination)
{
  if (node && node[key]) {
    destination = node[key].as<T>();
  }
}

[[nodiscard]] bool finiteNonNegative(double value) noexcept
{
  return std::isfinite(value) && value >= 0.0;
}

[[nodiscard]] bool finitePositive(double value) noexcept
{
  return std::isfinite(value) && value > 0.0;
}

void validate(const PlannerTuning& tuning)
{
  const PlannerConfig& config = tuning.planner;
  const ArmorScoreWeights& weights = tuning.armor_score_weights;
  const double weight_sum =
    weights.facing_weight + weights.window_weight + weights.aim_cost_weight;

  const bool base_valid =
    tuning.latency.ready()
    && finiteNonNegative(tuning.facing_angle_good)
    && finitePositive(tuning.facing_angle_bad)
    && tuning.facing_angle_bad > tuning.facing_angle_good
    && config.max_iterations > 0
    && config.fly_time_tolerance.count() > 0
    && finitePositive(config.position_tolerance)
    && finiteNonNegative(config.angle_tolerance)
    && finitePositive(config.gravity)
    && (!config.enable_air_resistance
        || finitePositive(config.linear_drag_coefficient))
    && finiteNonNegative(config.switch_dead_zone)
    && finiteNonNegative(config.score_switch_threshold)
    && config.score_switch_stable_frames > 0
    && finiteNonNegative(config.rotation_rate_dead_zone)
    && config.lock_stable_frames > 0
    && finiteNonNegative(config.aim_cost_good_angle)
    && finitePositive(config.aim_cost_bad_angle)
    && config.aim_cost_bad_angle > config.aim_cost_good_angle
    && finitePositive(config.normal_enter_angle)
    && finiteNonNegative(config.normal_leave_angle)
    && config.normal_leave_angle <= config.normal_enter_angle
    && finitePositive(config.outpost_enter_angle)
    && finiteNonNegative(config.outpost_leave_angle)
    && config.outpost_leave_angle <= config.outpost_enter_angle
    && config.max_lost_frames >= 0
    && finiteNonNegative(weights.facing_weight)
    && finiteNonNegative(weights.window_weight)
    && finiteNonNegative(weights.aim_cost_weight)
    && std::abs(weight_sum - 1.0) <= 1e-9;

  const bool mpc_valid =
    !config.enable_mpc
    || (finiteNonNegative(config.yaw_angle_weight)
        && finiteNonNegative(config.yaw_velocity_weight)
        && finitePositive(config.yaw_acceleration_weight)
        && finiteNonNegative(config.pitch_angle_weight)
        && finiteNonNegative(config.pitch_velocity_weight)
        && finitePositive(config.pitch_acceleration_weight)
        && std::isfinite(config.min_yaw_acceleration)
        && std::isfinite(config.max_yaw_acceleration)
        && config.min_yaw_acceleration <= config.max_yaw_acceleration
        && std::isfinite(config.min_pitch_acceleration)
        && std::isfinite(config.max_pitch_acceleration)
        && config.min_pitch_acceleration <= config.max_pitch_acceleration
        && config.mpc_max_iterations > 0
        && finitePositive(config.mpc_admm_rho)
        && finitePositive(config.mpc_primal_tolerance)
        && finitePositive(config.mpc_dual_tolerance));

  if (!base_valid || !mpc_valid) {
    throw std::invalid_argument(
      "planner configuration contains invalid values or inconsistent ranges");
  }
}

}  // namespace

PlannerTuning loadPlannerTuning(const std::string& config_path)
{
  const YAML::Node root = YAML::LoadFile(config_path);
  PlannerTuning tuning;
  PlannerConfig& config = tuning.planner;

  const YAML::Node latency = root["latency"];
  readOptional(latency, "fire_delay_s", tuning.latency.fire_delay);

  const YAML::Node prediction = root["prediction"];
  readOptional(prediction, "max_iterations", config.max_iterations);
  int fly_time_tolerance_us =
    static_cast<int>(config.fly_time_tolerance.count());
  readOptional(prediction, "fly_time_tolerance_us", fly_time_tolerance_us);
  config.fly_time_tolerance =
    std::chrono::microseconds{fly_time_tolerance_us};
  readOptional(
    prediction, "position_tolerance_m", config.position_tolerance);
  readOptional(prediction, "angle_tolerance_rad", config.angle_tolerance);

  const YAML::Node ballistic = root["ballistic"];
  readOptional(ballistic, "gravity_mps2", config.gravity);
  readOptional(
    ballistic, "enable_air_resistance", config.enable_air_resistance);
  readOptional(
    ballistic,
    "linear_drag_coefficient_per_s",
    config.linear_drag_coefficient);

  const YAML::Node selection = root["selection"];
  readOptional(selection, "switch_dead_zone_deg", config.switch_dead_zone);
  readOptional(
    selection, "score_switch_threshold", config.score_switch_threshold);
  readOptional(
    selection,
    "score_switch_stable_frames",
    config.score_switch_stable_frames);
  readOptional(
    selection,
    "rotation_rate_dead_zone_rad_s",
    config.rotation_rate_dead_zone);
  readOptional(selection, "lock_stable_frames", config.lock_stable_frames);
  readOptional(selection, "max_lost_frames", config.max_lost_frames);
  readOptional(
    selection, "aim_cost_good_angle_deg", config.aim_cost_good_angle);
  readOptional(
    selection, "aim_cost_bad_angle_deg", config.aim_cost_bad_angle);
  readOptional(
    selection, "normal_enter_angle_deg", config.normal_enter_angle);
  readOptional(
    selection, "normal_leave_angle_deg", config.normal_leave_angle);
  readOptional(
    selection, "outpost_enter_angle_deg", config.outpost_enter_angle);
  readOptional(
    selection, "outpost_leave_angle_deg", config.outpost_leave_angle);
  readOptional(
    selection, "facing_angle_good_deg", tuning.facing_angle_good);
  readOptional(
    selection, "facing_angle_bad_deg", tuning.facing_angle_bad);

  const YAML::Node score_weights = selection["score_weights"];
  readOptional(
    score_weights, "facing", tuning.armor_score_weights.facing_weight);
  readOptional(
    score_weights, "window", tuning.armor_score_weights.window_weight);
  readOptional(
    score_weights, "aim_cost", tuning.armor_score_weights.aim_cost_weight);

  const YAML::Node mpc = root["mpc"];
  readOptional(mpc, "enable", config.enable_mpc);
  readOptional(mpc, "max_iterations", config.mpc_max_iterations);
  readOptional(mpc, "admm_rho", config.mpc_admm_rho);
  readOptional(mpc, "primal_tolerance", config.mpc_primal_tolerance);
  readOptional(mpc, "dual_tolerance", config.mpc_dual_tolerance);

  const YAML::Node yaw = mpc["yaw"];
  readOptional(yaw, "angle_weight", config.yaw_angle_weight);
  readOptional(yaw, "velocity_weight", config.yaw_velocity_weight);
  readOptional(
    yaw, "acceleration_weight", config.yaw_acceleration_weight);
  readOptional(
    yaw, "min_acceleration_rad_s2", config.min_yaw_acceleration);
  readOptional(
    yaw, "max_acceleration_rad_s2", config.max_yaw_acceleration);

  const YAML::Node pitch = mpc["pitch"];
  readOptional(pitch, "angle_weight", config.pitch_angle_weight);
  readOptional(pitch, "velocity_weight", config.pitch_velocity_weight);
  readOptional(
    pitch, "acceleration_weight", config.pitch_acceleration_weight);
  readOptional(
    pitch, "min_acceleration_rad_s2", config.min_pitch_acceleration);
  readOptional(
    pitch, "max_acceleration_rad_s2", config.max_pitch_acceleration);

  validate(tuning);
  return tuning;
}

}  // namespace L4Planning
