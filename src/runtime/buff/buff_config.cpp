#include "runtime/buff/buff_config.hpp"

#include "l6_telemetry/logger.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace runtime::buff {
namespace {

template <typename T>
void read(const YAML::Node& node, const char* key, T& value)
{
  if (!node || !node[key]) return;
  try {
    value = node[key].as<T>();
  } catch (const YAML::Exception& error) {
    L6Telemetry::logWarn("buff config invalid field", key, error.what());
  }
}

void readMilliseconds(const YAML::Node& node, const char* key, double& seconds)
{
  double milliseconds = seconds * 1000.0;
  read(node, key, milliseconds);
  seconds = milliseconds / 1000.0;
}

template <typename T>
void positiveOrDefault(T& value, T fallback)
{
  if (!std::isfinite(value) || value <= 0) value = fallback;
}

}  // namespace

Config loadConfig(const std::string& path)
{
  Config config;
  const YAML::Node root = YAML::LoadFile(path);
  const auto detector = root["detector"];
  std::string method = "traditional";
  read(detector, "method", method);
  if (method != "traditional" && method != "model") {
    throw std::invalid_argument("buff detector.method must be traditional or model");
  }
  config.detector.use_model = method == "model";
  const auto inference = detector ? detector["inference"] : YAML::Node{};
  if (inference) {
    std::string path = config.detector.model.model_path.string();
    read(inference, "model_path", path);
    config.detector.model.model_path = path;
    read(inference, "device", config.detector.model.device);
    std::string color = "rgb";
    read(inference, "model_color_order", color);
    if (color != "rgb" && color != "bgr") {
      throw std::invalid_argument("buff model_color_order must be rgb or bgr");
    }
    config.detector.model.model_color_order = color == "rgb"
      ? L2Perception::ModelColorOrder::Rgb : L2Perception::ModelColorOrder::Bgr;
    read(inference, "confidence_threshold", config.detector.decoder.confidence_threshold);
    read(inference, "nms_iou_threshold", config.detector.decoder.nms_iou_threshold);
  }
  // 提前验证阈值，让配置错误在启动时暴露。
  (void)L2Perception::Buff::Decoder{config.detector.decoder};
  read(detector, "min_distance_m", config.detector.min_distance);
  read(detector, "max_distance_m", config.detector.max_distance);
  read(detector, "max_perspective_deg", config.detector.max_perspective_deg);
  read(detector, "active_threshold", config.detector.active_threshold);
  read(detector, "match_threshold", config.detector.match_threshold);

  const auto tracker = root["tracker"];
  read(tracker, "noise_x", config.tracker.noise_x);
  read(tracker, "noise_y", config.tracker.noise_y);
  read(tracker, "noise_z", config.tracker.noise_z);
  read(tracker, "noise_rotation_angle", config.tracker.noise_rotation_angle);
  read(tracker, "noise_rotation_speed", config.tracker.noise_rotation_speed);
  read(tracker, "noise_face_yaw", config.tracker.noise_face_yaw);
  read(tracker, "noise_observation", config.tracker.noise_observation);
  read(tracker, "gate_threshold", config.tracker.gate_threshold);
  read(tracker, "init_seed_mean_error", config.tracker.init_seed_mean_error);
  read(tracker, "init_seed_max_error", config.tracker.init_seed_max_error);
  read(tracker, "init_center_gate", config.tracker.init_center_gate);
  read(tracker, "init_pitch_bound_deg", config.tracker.init_pitch_bound_deg);
  read(tracker, "diverge_face_angle_deg", config.tracker.diverge_face_angle_deg);
  read(tracker, "timeout_seconds", config.tracker.timeout_seconds);
  readMilliseconds(tracker, "inactive_timeout_ms", config.tracker.inactive_timeout_seconds);

  const auto planning = root["planning"];
  // 兼容迁移初期的 bullet_speed_mps；新字段若存在则覆盖旧字段。
  read(planning, "bullet_speed_mps", config.planner.fallback_bullet_speed);
  read(planning, "fallback_bullet_speed_mps", config.planner.fallback_bullet_speed);
  read(planning, "min_valid_bullet_speed_mps", config.planner.min_valid_bullet_speed);
  read(planning, "trust_fallback_bullet_speed", config.planner.trust_fallback_bullet_speed);
  readMilliseconds(planning, "shoot_delay_ms", config.planner.shoot_delay);
  read(planning, "yaw_offset_deg", config.planner.offset_yaw_deg);
  read(planning, "pitch_offset_deg", config.planner.offset_pitch_deg);
  readMilliseconds(planning, "idle_duration_ms", config.planner.idle_duration);
  readMilliseconds(planning, "shoot_duration_ms", config.planner.shoot_duration);

  read(root["control"], "shoot_enable", config.control.shoot_enable);
  read(root["debug"], "overlay", config.overlay);
  read(root["debug"], "plot", config.plot);

  positiveOrDefault(config.detector.min_distance, 1.7);
  positiveOrDefault(config.detector.max_distance, 5.0);
  if (!std::isfinite(config.detector.max_perspective_deg) ||
      config.detector.max_perspective_deg <= 0.0 ||
      config.detector.max_perspective_deg >= 90.0) {
    config.detector.max_perspective_deg = 60.0;
  }
  positiveOrDefault(config.tracker.noise_observation, 20.0);
  positiveOrDefault(config.tracker.gate_threshold, 13.816);
  positiveOrDefault(config.tracker.timeout_seconds, 1.5);
  positiveOrDefault(config.tracker.inactive_timeout_seconds, 0.1);
  positiveOrDefault(config.planner.fallback_bullet_speed, 22.5);
  positiveOrDefault(config.planner.min_valid_bullet_speed, 14.0);
  positiveOrDefault(config.planner.shoot_delay, 0.04);
  positiveOrDefault(config.planner.idle_duration, 0.4);
  positiveOrDefault(config.planner.shoot_duration, 0.2);
  if (config.detector.max_distance <= config.detector.min_distance) {
    config.detector.max_distance = 5.0;
  }
  return config;
}

}  // namespace runtime::buff
