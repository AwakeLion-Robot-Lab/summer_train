#include "l5_control/fire_decision.hpp"

#include "l6_telemetry/logger.hpp"
#include "yaml.hpp"

#include <chrono>
#include <optional>
#include <string>

namespace L5Control {
namespace {

constexpr double kPi = 3.14159265358979323846;

double degreeToRadian(double degree) noexcept
{
  return degree * kPi / 180.0;
}

std::optional<double> readOptionalDouble(
  const YAML::Node& yaml,
  const std::string& key)
{
  if (!yaml[key]) {
    return std::nullopt;
  }

  try {
    return yaml[key].as<double>();
  } catch (const YAML::Exception& error) {
    L6Telemetry::logWarn("fire config invalid field", key, error.what());
    return std::nullopt;
  }
}

std::optional<double> readOptionalAngle(
  const YAML::Node& yaml,
  const std::string& key)
{
  const auto degree = readOptionalDouble(yaml, key);
  if (!degree.has_value()) {
    return std::nullopt;
  }

  return degreeToRadian(*degree);
}

int readOptionalInt(
  const YAML::Node& yaml,
  const std::string& key,
  int default_value)
{
  if (!yaml[key]) {
    return default_value;
  }

  try {
    return yaml[key].as<int>();
  } catch (const YAML::Exception& error) {
    L6Telemetry::logWarn("fire config invalid field", key, error.what());
    return default_value;
  }
}

bool readOptionalBool(
  const YAML::Node& yaml,
  const std::string& key,
  bool default_value)
{
  if (!yaml[key]) {
    return default_value;
  }

  try {
    return yaml[key].as<bool>();
  } catch (const YAML::Exception& error) {
    L6Telemetry::logWarn("fire config invalid field", key, error.what());
    return default_value;
  }
}

}  // namespace

FireConfig loadFireConfig(const std::string& config_path)
{
  FireConfig config;
  const YAML::Node yaml = tools::load(config_path);

  config.shoot_enable =
    readOptionalBool(yaml, "shoot_enable", config.shoot_enable);
  config.bullet_diameter = readOptionalDouble(yaml, "bullet_diameter_m");
  config.min_bullet_speed = readOptionalDouble(yaml, "min_bullet_speed_mps");
  config.max_bullet_speed = readOptionalDouble(yaml, "max_bullet_speed_mps");
  config.heat_limit = readOptionalDouble(yaml, "heat_limit");

  config.min_yaw = readOptionalAngle(yaml, "min_yaw_deg");
  config.max_yaw = readOptionalAngle(yaml, "max_yaw_deg");
  config.min_pitch = readOptionalAngle(yaml, "min_pitch_deg");
  config.max_pitch = readOptionalAngle(yaml, "max_pitch_deg");

  config.yaw_distance_boundary =
    readOptionalDouble(yaml, "yaw_distance_boundary_m");
  config.near_max_yaw_command_jump =
    readOptionalAngle(yaml, "near_max_yaw_command_jump_deg");
  config.near_max_yaw_error =
    readOptionalAngle(yaml, "near_max_yaw_error_deg");
  config.far_max_yaw_command_jump =
    readOptionalAngle(yaml, "far_max_yaw_command_jump_deg");
  config.far_max_yaw_error =
    readOptionalAngle(yaml, "far_max_yaw_error_deg");

  config.max_aim_pitch_error =
    readOptionalAngle(yaml, "max_aim_pitch_error_deg");
  config.max_pitch_command_jump =
    readOptionalAngle(yaml, "max_pitch_command_jump_deg");
  config.max_pitch_error =
    readOptionalAngle(yaml, "max_pitch_error_deg");

  config.max_robot_state_age = std::chrono::milliseconds{
    readOptionalInt(
      yaml,
      "max_robot_state_age_ms",
      static_cast<int>(config.max_robot_state_age.count()))};
  config.max_gimbal_pose_age = std::chrono::milliseconds{
    readOptionalInt(
      yaml,
      "max_gimbal_pose_age_ms",
      static_cast<int>(config.max_gimbal_pose_age.count()))};
  config.max_plan_age = std::chrono::milliseconds{
    readOptionalInt(
      yaml,
      "max_plan_age_ms",
      static_cast<int>(config.max_plan_age.count()))};

  return config;
}

}  // namespace L5Control
