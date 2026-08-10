#include "l1_sensor/talos/talos_config.hpp"

#include <string>

#include <yaml-cpp/yaml.h>

namespace L1Sensor::talos {
namespace {

EnemyColor parseEnemyColor(const std::string& value)
{
  if (value == "red" || value == "Red") {
    return EnemyColor::Red;
  }
  if (value == "blue" || value == "Blue") {
    return EnemyColor::Blue;
  }
  return EnemyColor::Unknown;
}

WorkMode parseWorkMode(const std::string& value)
{
  if (value == "outpost") {
    return WorkMode::Outpost;
  }
  if (value == "small_buff" || value == "small-buff") {
    return WorkMode::SmallBuff;
  }
  if (value == "big_buff" || value == "big-buff") {
    return WorkMode::BigBuff;
  }
  if (value == "idle") {
    return WorkMode::Idle;
  }
  return WorkMode::AutoAim;
}

}  // namespace

TalosSimConfig loadTalosSimConfig(const std::string& config_path)
{
  TalosSimConfig config;
  const YAML::Node root = YAML::LoadFile(config_path);
  if (const YAML::Node calibration = root["calibration"]) {
    config.calibration =
      L1Sensor::loadCameraCalibration(calibration, config_path + ": calibration");
  }
  const YAML::Node node = root["talos"];
  if (!node) {
    return config;
  }
  if (node["shm_dir"]) {
    config.shm_dir = node["shm_dir"].as<std::string>();
  }
  if (node["enemy"]) {
    config.enemy_color = parseEnemyColor(node["enemy"].as<std::string>());
  }
  if (node["mode"]) {
    config.mode = parseWorkMode(node["mode"].as<std::string>());
  }
  if (node["bullet_speed"]) {
    config.bullet_speed = node["bullet_speed"].as<double>();
  }
  if (node["shoot_enable"]) {
    config.shoot_enable = node["shoot_enable"].as<bool>();
  }
  if (node["timeout_ms"]) {
    config.timeout = std::chrono::milliseconds{node["timeout_ms"].as<int>()};
  }
  if (node["default_distance_m"]) {
    config.default_distance_m = node["default_distance_m"].as<double>();
  }
  if (node["gt_match_gate_m"]) {
    config.gt_match_gate_m = node["gt_match_gate_m"].as<double>();
  }
  if (node["report_interval_s"]) {
    config.report_interval_s = node["report_interval_s"].as<double>();
  }
  return config;
}

}  // namespace L1Sensor::talos
