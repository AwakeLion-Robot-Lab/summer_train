#include "l1_sensor/talos/talos_config.hpp"

#include <yaml-cpp/yaml.h>

namespace L1Sensor::talos {
namespace {
EnemyColor enemyColor(const std::string& value)
{
  if (value == "red" || value == "Red") return EnemyColor::Red;
  if (value == "blue" || value == "Blue") return EnemyColor::Blue;
  return EnemyColor::Unknown;
}

WorkMode workMode(const std::string& value)
{
  if (value == "idle") return WorkMode::Idle;
  if (value == "outpost") return WorkMode::Outpost;
  if (value == "small_buff" || value == "small-buff") return WorkMode::SmallBuff;
  if (value == "big_buff" || value == "big-buff") return WorkMode::BigBuff;
  return WorkMode::AutoAim;
}
} // namespace

TalosConfig loadTalosConfig(const std::string& path)
{
  TalosConfig config;
  const YAML::Node root = YAML::LoadFile(path);
  if (const auto calibration = root["calibration"]) {
    config.calibration = loadCameraCalibration(calibration, path + ": calibration");
  }
  const YAML::Node node = root["talos"];
  if (!node) return config;
  if (node["shm_dir"]) config.shm_dir = node["shm_dir"].as<std::string>();
  if (node["enemy"]) config.enemy_color = enemyColor(node["enemy"].as<std::string>());
  if (node["mode"]) config.mode = workMode(node["mode"].as<std::string>());
  if (node["bullet_speed"]) config.bullet_speed = node["bullet_speed"].as<double>();
  if (node["shoot_enable"]) config.shoot_enable = node["shoot_enable"].as<bool>();
  if (node["timeout_ms"]) config.timeout = std::chrono::milliseconds{node["timeout_ms"].as<int>()};
  if (node["default_distance_m"]) config.default_distance_m = node["default_distance_m"].as<double>();
  if (node["max_yaw_jump_deg"]) config.max_yaw_jump_deg = node["max_yaw_jump_deg"].as<double>();
  if (node["max_pitch_jump_deg"]) config.max_pitch_jump_deg = node["max_pitch_jump_deg"].as<double>();
  return config;
}

} // namespace L1Sensor::talos
