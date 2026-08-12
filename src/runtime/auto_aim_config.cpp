#include "runtime/auto_aim_config.hpp"

#include "l6_telemetry/logger.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <filesystem>
#include <numbers>
#include <stdexcept>

namespace runtime {
namespace {

// 三个取值助手：键缺失就保留传入的默认值，因此 YAML 里只写想改的项即可。
template <typename T>
void assign(const YAML::Node& node, const char* key, T& value)
{
  if (node && node[key]) {
    value = node[key].as<T>();
  }
}

void assignDegrees(const YAML::Node& node, const char* key, double& radians)
{
  if (node && node[key]) {
    radians = node[key].as<double>() * std::numbers::pi / 180.0;
  }
}

void assignMilliseconds(const YAML::Node& node, const char* key, double& seconds)
{
  if (node && node[key]) {
    seconds = node[key].as<double>() / 1000.0;
  }
}

void assignOptionalDegrees(
  const YAML::Node& node, const char* key, std::optional<double>& radians)
{
  if (node && node[key]) {
    radians = node[key].as<double>() * std::numbers::pi / 180.0;
  }
}

template <typename T>
void assignOptional(const YAML::Node& node, const char* key, std::optional<T>& value)
{
  if (node && node[key]) {
    value = node[key].as<T>();
  }
}

void loadArmor(const YAML::Node& node, L3Estimation::ArmorConfig& armor)
{
  assign(node, "small_width", armor.small_width);
  assign(node, "big_width", armor.big_width);
  assign(node, "height", armor.height);
  assignDegrees(node, "mount_pitch_deg", armor.mount_pitch);
  assignDegrees(node, "outpost_mount_pitch_deg", armor.outpost_mount_pitch);
  assignDegrees(node, "yaw_search_range_deg", armor.yaw_search_range);
  assignDegrees(node, "yaw_coarse_step_deg", armor.yaw_coarse_step);
  assignDegrees(node, "yaw_fine_range_deg", armor.yaw_fine_range);
  assignDegrees(node, "yaw_fine_step_deg", armor.yaw_fine_step);
  assign(node, "yaw_cost_huber_px", armor.yaw_cost_huber_px);
  assign(node, "yaw_cost_huber_deg", armor.yaw_cost_huber_deg);
  assign(node, "yaw_cost_shape_weight", armor.yaw_cost_shape_weight);
}

void loadTracking(const YAML::Node& node, L3Estimation::TrackerConfig& tracker)
{
  assign(node, "min_detect_count", tracker.min_detect_count);
  if (node && node["max_frame_interval_ms"]) {
    tracker.max_frame_interval =
      std::chrono::milliseconds(node["max_frame_interval_ms"].as<int>());
  }
  assign(node, "max_temp_lost_count", tracker.max_temp_lost_count);
  assign(node, "outpost_max_temp_lost_count", tracker.outpost_max_temp_lost_count);
}

void loadGtsam(const YAML::Node& node, L3Estimation::GtsamEst::Config& gtsam)
{
  assign(node, "max_match_distance", gtsam.max_match_distance);
  assignDegrees(node, "max_match_yaw_diff_deg", gtsam.max_match_yaw_diff);
  assign(node, "first_update_batch_size", gtsam.first_update_batch_size);
  if (node && node["lost_threshold_ms"]) {
    gtsam.lost_threshold =
      std::chrono::milliseconds(node["lost_threshold_ms"].as<int>());
  }

  assign(node, "translation_prior_sigma", gtsam.translation_prior_sigma);
  assign(node, "velocity_prior_sigma", gtsam.velocity_prior_sigma);
  assign(node, "yaw_prior_sigma", gtsam.yaw_prior_sigma);
  assign(node, "vyaw_prior_sigma", gtsam.vyaw_prior_sigma);
  assign(node, "radius_prior_sigma", gtsam.radius_prior_sigma);
  assign(node, "dz_prior_sigma", gtsam.dz_prior_sigma);
  assign(node, "default_radius", gtsam.default_radius);
  assign(node, "radius_min", gtsam.radius_min);
  assign(node, "radius_max", gtsam.radius_max);
  assign(node, "default_dz", gtsam.default_dz);

  assign(node, "translation_factor_sigma", gtsam.translation_factor_sigma);
  assign(node, "velocity_factor_sigma", gtsam.velocity_factor_sigma);
  assign(node, "yaw_factor_sigma", gtsam.yaw_factor_sigma);
  assign(node, "vyaw_factor_sigma", gtsam.vyaw_factor_sigma);

  assign(node, "obs_tangential_sigma", gtsam.obs_tangential_sigma);
  assign(node, "obs_radial_sigma", gtsam.obs_radial_sigma);
  assign(node, "obs_height_sigma", gtsam.obs_height_sigma);
  assign(node, "obs_yaw_sigma", gtsam.obs_yaw_sigma);
  assign(node, "obs_pixel_sigma", gtsam.obs_pixel_sigma);
}

void loadTarget(const YAML::Node& node, L3Estimation::TargetConfig& target)
{
  assign(node, "accel_variance", target.accel_variance);
  assign(node, "yaw_accel_variance", target.yaw_accel_variance);
  assign(node, "outpost_accel_variance", target.outpost_accel_variance);
  assign(node, "outpost_yaw_accel_variance", target.outpost_yaw_accel_variance);
  assign(node, "radius", target.radius);
  assign(node, "outpost_radius", target.outpost_radius);
  assign(node, "base_radius", target.base_radius);
  assign(node, "min_radius", target.min_radius);
  assign(node, "max_radius", target.max_radius);
}

void loadFilter(
  const YAML::Node& node,
  L3Estimation::FilterEst::TargetConfig& filter)
{
  assign(node, "angle_variance", filter.angle_variance);
  assign(node, "distance_variance", filter.distance_variance);
  assign(node, "armor_yaw_variance", filter.armor_yaw_variance);
  assign(node, "min_update_count", filter.min_update_count);
  assign(node, "outpost_min_update_count", filter.outpost_min_update_count);
  assign(node, "max_nis_failure_ratio", filter.max_nis_failure_ratio);
  assign(node, "outpost_v_yaw", filter.outpost_v_yaw);
}

void loadPlanning(const YAML::Node& node, L4Planning::PlanConfig& plan)
{
  assign(node, "max_iterations", plan.max_iterations);
  if (node && node["fly_time_tolerance_ms"]) {
    plan.fly_time_tolerance = std::chrono::microseconds(
      static_cast<int>(node["fly_time_tolerance_ms"].as<double>() * 1000.0));
  }
  assignMilliseconds(node, "high_speed_delay_ms", plan.high_speed_delay_time);
  assignMilliseconds(node, "low_speed_delay_ms", plan.low_speed_delay_time);
  assign(node, "decision_speed", plan.decision_speed);
  assignDegrees(node, "yaw_offset_deg", plan.yaw_offset);
  assignDegrees(node, "pitch_offset_deg", plan.pitch_offset);
  assign(node, "fallback_bullet_speed", plan.fallback_bullet_speed);
  assign(node, "min_valid_bullet_speed", plan.min_valid_bullet_speed);
  if (node && node["send_to_control_ms"]) {
    plan.send_to_control = node["send_to_control_ms"].as<double>() / 1000.0;
  }
  if (node && node["control_to_fire_ms"]) {
    plan.control_to_fire = node["control_to_fire_ms"].as<double>() / 1000.0;
  }

  if (!node) {
    return;
  }
  const YAML::Node ballistic = node["ballistic"];
  assign(ballistic, "gravity", plan.ballistic.gravity);
  assign(ballistic, "drag_coefficient", plan.ballistic.drag_coefficient);
  assignDegrees(ballistic, "max_pitch_deg", plan.ballistic.max_pitch);

  const YAML::Node selector = node["selector"];
  assignDegrees(selector, "front_window_deg", plan.selector.front_window);
  assignDegrees(selector, "outpost_coming_deg", plan.selector.outpost_coming_angle);
  assignDegrees(selector, "outpost_leaving_deg", plan.selector.outpost_leaving_angle);
}

void loadFire(const YAML::Node& node, L5Control::FireConfig& fire)
{
  assign(node, "shoot_enable", fire.shoot_enable);
  assignOptional(node, "bullet_diameter", fire.bullet_diameter);
  assignOptional(node, "min_bullet_speed", fire.min_bullet_speed);
  assignOptional(node, "max_bullet_speed", fire.max_bullet_speed);
  assignOptional(node, "heat_limit", fire.heat_limit);
  assignOptionalDegrees(node, "min_yaw_deg", fire.min_yaw);
  assignOptionalDegrees(node, "max_yaw_deg", fire.max_yaw);
  assignOptionalDegrees(node, "min_pitch_deg", fire.min_pitch);
  assignOptionalDegrees(node, "max_pitch_deg", fire.max_pitch);
  if (node && node["max_robot_state_age_ms"]) {
    fire.max_robot_state_age =
      std::chrono::milliseconds(node["max_robot_state_age_ms"].as<int>());
  }
  if (node && node["max_gimbal_pose_age_ms"]) {
    fire.max_gimbal_pose_age =
      std::chrono::milliseconds(node["max_gimbal_pose_age_ms"].as<int>());
  }
  assign(node, "armor_width_small", fire.armor_width_small);
  assign(node, "armor_width_big", fire.armor_width_big);
  assign(node, "armor_height", fire.armor_height);
  assign(node, "hit_margin_ratio", fire.hit_margin_ratio);
  assignDegrees(node, "min_yaw_tolerance_deg", fire.min_yaw_tolerance);
  assignDegrees(node, "min_pitch_tolerance_deg", fire.min_pitch_tolerance);
}

}  // namespace

AutoAimConfig loadAutoAimConfig(const std::string& path)
{
  AutoAimConfig config;
  if (!std::filesystem::exists(path)) {
    L6Telemetry::logWarn("auto-aim config not found, using defaults", path);
    return config;
  }

  const YAML::Node root = YAML::LoadFile(path);

  const YAML::Node inference = root["inference"];
  if (inference && inference["model_path"]) {
    config.model_path = inference["model_path"].as<std::string>();
  }
  assign(inference, "device", config.inference_device);
  if (inference && inference["backend"]) {
    const std::string name = inference["backend"].as<std::string>();
    const auto backend = L2Perception::inferenceBackendFromString(name);
    if (!backend) {
      throw std::runtime_error(
        "inference.backend must be 'openvino' or 'tensorrt'; got " + name);
    }
    config.inference_backend = *backend;
  }

  // 后端名写错时保留默认的 filter 并告警：估计器选错和噪声参数填错一样致命，
  // 但这里静默用默认值比让整条链路起不来更符合本项目的降级约定。
  const YAML::Node estimator = root["estimator"];
  if (estimator && estimator["backend"]) {
    const auto name = estimator["backend"].as<std::string>();
    if (const auto backend = L3Estimation::estimatorBackendFromString(name)) {
      config.estimator = *backend;
    } else {
      L6Telemetry::logWarn("unknown estimator backend, falling back to filter:", name);
    }
  }

  loadArmor(root["armor"], config.armor);
  loadTracking(root["tracking"], config.tracker);
  loadTarget(root["target"], config.target);
  loadFilter(root["filter"], config.filter);
  loadGtsam(root["gtsam"], config.gtsam);
  loadPlanning(root["planning"], config.plan);
  loadFire(root["fire"], config.fire);

  L6Telemetry::logInfo("auto-aim config loaded", path);
  if (config.fire.shoot_enable) {
    L6Telemetry::logWarn("firing is ENABLED by config");
  }
  return config;
}

}  // namespace runtime
