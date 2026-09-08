#include "runtime/auto_aim_config.hpp"

#include "l6_telemetry/logger.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <numbers>
#include <stdexcept>
#include <string>

namespace runtime {
namespace {

template <typename T>
void readValue(
  const YAML::Node& section,
  const char* key,
  T& value)
{
  if (!section || !section[key]) {
    return;
  }

  try {
    value = section[key].as<T>();
  } catch (const YAML::Exception& error) {
    L6Telemetry::logWarn("auto-aim config invalid field", key, error.what());
  }
}

void readMilliseconds(
  const YAML::Node& section,
  const char* key,
  std::chrono::milliseconds& value)
{
  int milliseconds = static_cast<int>(value.count());
  readValue(section, key, milliseconds);
  value = std::chrono::milliseconds{milliseconds};
}

void readMicroseconds(
  const YAML::Node& section,
  const char* key,
  std::chrono::microseconds& value)
{
  int microseconds = static_cast<int>(value.count());
  readValue(section, key, microseconds);
  value = std::chrono::microseconds{microseconds};
}

void readMillisecondsAsSeconds(
  const YAML::Node& section,
  const char* key,
  double& seconds)
{
  double milliseconds = seconds * 1e3;
  readValue(section, key, milliseconds);
  seconds = milliseconds * 1e-3;
}

void readDegrees(
  const YAML::Node& section,
  const char* key,
  double& radians)
{
  double degrees = radians * 180.0 / std::numbers::pi;
  readValue(section, key, degrees);
  radians = degrees * std::numbers::pi / 180.0;
}

bool positiveFinite(double value) noexcept
{
  return std::isfinite(value) && value > 0.0;
}

void normalize(AutoAimConfig& config)
{
  const L3Estimation::ArmorConfig armor_defaults;
  if (!positiveFinite(config.armor.small_width)) {
    config.armor.small_width = armor_defaults.small_width;
  }
  if (!positiveFinite(config.armor.big_width)) {
    config.armor.big_width = armor_defaults.big_width;
  }
  if (!positiveFinite(config.armor.height)) {
    config.armor.height = armor_defaults.height;
  }
  if (!positiveFinite(config.armor.corner_noise_px)) {
    config.armor.corner_noise_px = armor_defaults.corner_noise_px;
  }

  config.tracker.min_detect_count =
    std::max(config.tracker.min_detect_count, 1);
  config.tracker.max_frame_interval = std::max(
    config.tracker.max_frame_interval, std::chrono::milliseconds{1});
  config.tracker.max_temp_lost_count =
    std::max(config.tracker.max_temp_lost_count, 0);
  config.tracker.outpost_max_temp_lost_count =
    std::max(config.tracker.outpost_max_temp_lost_count, 0);

  // 精修参数越界会让它静默失效（阈值 255 时二值图全黑，一块灯条也找不到）
  // 或者全盘接受（端点距离无穷大时任何传统解都覆盖网络角点），都不会报错。
  const L2Perception::ArmorRefinerConfig refiner_defaults;
  if (!(config.refiner.binary_threshold > 0.0 &&
        config.refiner.binary_threshold < 255.0)) {
    config.refiner.binary_threshold = refiner_defaults.binary_threshold;
  }
  if (!positiveFinite(config.refiner.min_lightbar_length_px)) {
    config.refiner.min_lightbar_length_px = refiner_defaults.min_lightbar_length_px;
  }
  if (!positiveFinite(config.refiner.max_endpoint_distance_px)) {
    config.refiner.max_endpoint_distance_px = refiner_defaults.max_endpoint_distance_px;
  }
  if (!(config.refiner.min_lightbar_ratio > 0.0F &&
        config.refiner.min_lightbar_ratio < config.refiner.max_lightbar_ratio)) {
    config.refiner.min_lightbar_ratio = refiner_defaults.min_lightbar_ratio;
    config.refiner.max_lightbar_ratio = refiner_defaults.max_lightbar_ratio;
  }
  if (!(config.refiner.max_angle_error_deg > 0.0F &&
        config.refiner.max_angle_error_deg <= 90.0F)) {
    config.refiner.max_angle_error_deg = refiner_defaults.max_angle_error_deg;
  }

  // 噪声为零或负会让 EKF 的增益直接发散，越界一律退回默认。
  const L3Estimation::TargetConfig target_defaults;
  for (const auto & [value, fallback] : {
         std::pair{&config.target.q_translation, target_defaults.q_translation},
         std::pair{&config.target.q_rotation, target_defaults.q_rotation},
         std::pair{&config.target.outpost_q_translation, target_defaults.outpost_q_translation},
         std::pair{&config.target.outpost_q_rotation, target_defaults.outpost_q_rotation},
         std::pair{&config.target.angle_variance, target_defaults.angle_variance},
         std::pair{&config.target.distance_variance_factor,
                   target_defaults.distance_variance_factor},
         std::pair{&config.target.armor_yaw_variance_base,
                   target_defaults.armor_yaw_variance_base},
         std::pair{&config.target.armor_yaw_distance_divisor,
                   target_defaults.armor_yaw_distance_divisor}}) {
    if (!positiveFinite(*value)) {
      *value = fallback;
    }
  }

  const L4Planning::ArmorPlanConfig plan_defaults;
  config.plan.impact.max_iterations = std::max(config.plan.impact.max_iterations, 1);
  config.plan.impact.fly_time_tolerance = std::max(
    config.plan.impact.fly_time_tolerance, std::chrono::microseconds{1});
  if (!std::isfinite(config.plan.impact.high_speed_delay_time) ||
      config.plan.impact.high_speed_delay_time < 0.0) {
    config.plan.impact.high_speed_delay_time = plan_defaults.impact.high_speed_delay_time;
  }
  if (!std::isfinite(config.plan.impact.low_speed_delay_time) ||
      config.plan.impact.low_speed_delay_time < 0.0) {
    config.plan.impact.low_speed_delay_time = plan_defaults.impact.low_speed_delay_time;
  }
  if (!std::isfinite(config.plan.impact.decision_speed) ||
      config.plan.impact.decision_speed < 0.0) {
    config.plan.impact.decision_speed = plan_defaults.impact.decision_speed;
  }
  if (!std::isfinite(config.plan.impact.yaw_offset)) {
    config.plan.impact.yaw_offset = plan_defaults.impact.yaw_offset;
  }
  if (!std::isfinite(config.plan.impact.pitch_offset)) {
    config.plan.impact.pitch_offset = plan_defaults.impact.pitch_offset;
  }
  if (!positiveFinite(config.plan.impact.fallback_bullet_speed)) {
    config.plan.impact.fallback_bullet_speed = plan_defaults.impact.fallback_bullet_speed;
  }
  if (!positiveFinite(config.plan.impact.min_valid_bullet_speed)) {
    config.plan.impact.min_valid_bullet_speed = plan_defaults.impact.min_valid_bullet_speed;
  }
  if (!positiveFinite(config.plan.selector.coming_angle)) {
    config.plan.selector.coming_angle = plan_defaults.selector.coming_angle;
  }
  if (!positiveFinite(config.plan.selector.leaving_angle)) {
    config.plan.selector.leaving_angle = plan_defaults.selector.leaving_angle;
  }
  if (!positiveFinite(config.plan.selector.outpost_coming_angle)) {
    config.plan.selector.outpost_coming_angle =
      plan_defaults.selector.outpost_coming_angle;
  }
  if (!positiveFinite(config.plan.selector.outpost_leaving_angle)) {
    config.plan.selector.outpost_leaving_angle =
      plan_defaults.selector.outpost_leaving_angle;
  }
  // 标定值必须是正的有限数，否则当作没标定。
  // 弹道参数越界都是静默失效：重力为负会解出朝天的仰角，阻力系数为负会让
  // 等效距离随距离指数缩短（越远打得越准，明显是错的），max_pitch 超过 90°
  // 则失去保护意义。三者任一非法就退回结构体默认值。
  const L4Planning::BallisticConfig ballistic_defaults;
  if (!positiveFinite(config.plan.ballistic.gravity)) {
    config.plan.ballistic.gravity = ballistic_defaults.gravity;
  }
  if (!(std::isfinite(config.plan.ballistic.drag_coefficient) &&
        config.plan.ballistic.drag_coefficient >= 0.0)) {
    L6Telemetry::logWarn("ballistic.drag_coefficient is invalid, using vacuum");
    config.plan.ballistic.drag_coefficient = ballistic_defaults.drag_coefficient;
  }
  if (!(config.plan.ballistic.max_pitch > 0.0 &&
        config.plan.ballistic.max_pitch < std::numbers::pi / 2.0)) {
    config.plan.ballistic.max_pitch = ballistic_defaults.max_pitch;
  }

  if (config.plan.impact.send_to_control &&
      !(std::isfinite(*config.plan.impact.send_to_control) &&
        *config.plan.impact.send_to_control >= 0.0)) {
    L6Telemetry::logWarn("planning.send_to_control_ms is invalid, treated as uncalibrated");
    config.plan.impact.send_to_control.reset();
  }

  const L5Control::FireConfig fire_defaults;
  if (!positiveFinite(config.fire.armor_width_small)) {
    config.fire.armor_width_small = fire_defaults.armor_width_small;
  }
  if (!positiveFinite(config.fire.armor_width_big)) {
    config.fire.armor_width_big = fire_defaults.armor_width_big;
  }
  if (!positiveFinite(config.fire.armor_height)) {
    config.fire.armor_height = fire_defaults.armor_height;
  }
  if (!std::isfinite(config.fire.hit_margin_ratio) ||
      config.fire.hit_margin_ratio <= 0.0 ||
      config.fire.hit_margin_ratio > 1.0) {
    config.fire.hit_margin_ratio = fire_defaults.hit_margin_ratio;
  }
  if (!positiveFinite(config.fire.min_yaw_tolerance)) {
    config.fire.min_yaw_tolerance = fire_defaults.min_yaw_tolerance;
  }
  if (!positiveFinite(config.fire.min_pitch_tolerance)) {
    config.fire.min_pitch_tolerance = fire_defaults.min_pitch_tolerance;
  }

  config.debug.overlay_every = std::max(config.debug.overlay_every, 1);

  const RuntimeSafetyConfig runtime_defaults;
  if (!std::isfinite(config.runtime.command_jump_threshold) ||
      config.runtime.command_jump_threshold < 0.0) {
    config.runtime.command_jump_threshold =
      runtime_defaults.command_jump_threshold;
  }
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
  if (!inference) {
    L6Telemetry::logWarn("auto-aim inference config missing, using defaults", path);
  } else {
    if (inference["backend"]) {
      const std::string name = inference["backend"].as<std::string>();
      const auto backend = L2Perception::inferenceBackendFromString(name);
      if (!backend) {
        throw std::runtime_error(
          "inference.backend must be 'openvino' or 'tensorrt'; got " + name);
      }
      config.inference_backend = *backend;
    }
    if (inference["model_path"]) {
      config.model_path = inference["model_path"].as<std::string>();
    }
    if (inference["device"]) {
      config.inference_device = inference["device"].as<std::string>();
    }

    // 后端调度参数。只影响编译期的算子调度，不改变数值结果——所以调这些
    // 不需要重新验证识别精度，但必须重新测帧耗时。
    //
    // 后三项只有 CPU 插件认识，device 是 GPU 时后端会静默跳过（GPU 插件收到
    // 会抛异常而不是忽略，过滤逻辑在 openvino_backend.cpp 里）。
    readValue(inference, "latency_hint", config.inference.latency_hint);
    readValue(inference, "num_threads", config.inference.inference_num_threads);
    if (inference["scheduling_core_type"]) {
      const std::string type = inference["scheduling_core_type"].as<std::string>();
      if (type == "any") {
        config.inference.scheduling_core_type = L2Perception::SchedulingCoreType::Any;
      } else if (type == "pcore") {
        config.inference.scheduling_core_type =
          L2Perception::SchedulingCoreType::PCoreOnly;
      } else if (type == "ecore") {
        config.inference.scheduling_core_type =
          L2Perception::SchedulingCoreType::ECoreOnly;
      } else {
        throw std::runtime_error(
          "inference.scheduling_core_type must be 'any', 'pcore' or 'ecore'; got " + type);
      }
    }
    // 缺省表示不下发该属性，保持 OpenVINO 默认；写 true/false 才显式指定。
    if (inference["enable_hyper_threading"]) {
      config.inference.enable_hyper_threading =
        inference["enable_hyper_threading"].as<bool>();
    }

    // 模型格式参数。这两项改错会直接改变推理结果，不是性能项：颜色顺序要和
    // 训练时一致，归一化除数要和导出时一致。宿主侧输入恒为 uint8 NHWC BGR。
    if (inference["model_color_order"]) {
      const std::string order = inference["model_color_order"].as<std::string>();
      if (order == "rgb" || order == "RGB") {
        config.inference.model_color_order = L2Perception::ModelColorOrder::Rgb;
      } else if (order == "bgr" || order == "BGR") {
        config.inference.model_color_order = L2Perception::ModelColorOrder::Bgr;
      } else {
        throw std::runtime_error(
          "inference.model_color_order must be 'rgb' or 'bgr'; got " + order);
      }
    }
    readValue(
      inference, "normalization_divisor", config.inference.normalization_divisor);
    if (!(config.inference.normalization_divisor > 0.0F)) {
      throw std::runtime_error("inference.normalization_divisor must be positive");
    }

    // 输出契约。layout 选定字段布局（下标由模型导出时定死，不在 YAML 里逐个
    // 手配），随后的阈值才是可调项。配错 layout 不会报错，只会解出垃圾角点。
    const YAML::Node decoder = inference["decoder"];
    if (decoder) {
      if (decoder["layout"]) {
        const std::string layout = decoder["layout"].as<std::string>();
        const auto preset = L2Perception::armorDecoderPreset(layout);
        if (!preset) {
          throw std::runtime_error(
            "inference.decoder.layout must be 'yolov5_22' or 'yolov8_21'; got " + layout);
        }
        config.decoder = *preset;
      }
      // 阈值在预设之后覆盖，顺序不能反。只有筛选策略可以从 YAML 调；
      // 字段布局属于 ArmorTensorContract，只能整组由 layout 选。
      readValue(decoder, "confidence_threshold", config.decoder.confidence_threshold);
      readValue(decoder, "minimum_confidence", config.decoder.minimum_confidence);
      readValue(decoder, "nms_iou_threshold", config.decoder.nms_iou_threshold);
      readValue(decoder, "nms_score_threshold", config.decoder.nms_score_threshold);
    }
  }

  // 三个单列字段是同一份配置的一部分，回填进去，构造 Backend 时只传一个结构体。
  config.inference.model_path = config.model_path;
  config.inference.device = config.inference_device;

  const YAML::Node armor = root["armor"];
  readValue(armor, "small_width_m", config.armor.small_width);
  readValue(armor, "big_width_m", config.armor.big_width);
  readValue(armor, "height_m", config.armor.height);
  readValue(armor, "corner_noise_px", config.armor.corner_noise_px);

  const YAML::Node tracker = root["tracker"];
  readValue(tracker, "min_detect_count", config.tracker.min_detect_count);
  readMilliseconds(
    tracker, "max_frame_interval_ms", config.tracker.max_frame_interval);
  readValue(
    tracker, "max_temp_lost_count", config.tracker.max_temp_lost_count);
  readValue(
    tracker,
    "outpost_max_temp_lost_count",
    config.tracker.outpost_max_temp_lost_count);

  // 传统灯条精修。默认值来自 sp_vision 的 standard3.yaml，按场地光照调
  // binary_threshold 是最常动的一个。
  const YAML::Node refiner = root["refiner"];
  readValue(refiner, "enable", config.refiner.enable);
  readValue(refiner, "binary_threshold", config.refiner.binary_threshold);
  readValue(refiner, "min_lightbar_length_px", config.refiner.min_lightbar_length_px);
  readValue(refiner, "max_angle_error_deg", config.refiner.max_angle_error_deg);
  readValue(refiner, "min_lightbar_ratio", config.refiner.min_lightbar_ratio);
  readValue(refiner, "max_lightbar_ratio", config.refiner.max_lightbar_ratio);
  readValue(
    refiner, "max_endpoint_distance_px", config.refiner.max_endpoint_distance_px);

  // EKF 噪声。回放标定的主要旋钮，改这些必须重跑 track_diag 看 NIS。
  const YAML::Node estimator = root["estimator"];
  readValue(estimator, "q_translation", config.target.q_translation);
  readValue(estimator, "q_rotation", config.target.q_rotation);
  readValue(estimator, "outpost_q_translation", config.target.outpost_q_translation);
  readValue(estimator, "outpost_q_rotation", config.target.outpost_q_rotation);
  readValue(estimator, "angle_variance", config.target.angle_variance);
  readValue(estimator, "distance_variance_factor", config.target.distance_variance_factor);
  readValue(estimator, "armor_yaw_variance_base", config.target.armor_yaw_variance_base);
  readValue(
    estimator, "armor_yaw_distance_divisor", config.target.armor_yaw_distance_divisor);

  const YAML::Node planning = root["planning"];
  readValue(planning, "max_iterations", config.plan.impact.max_iterations);
  readMicroseconds(
    planning, "fly_time_tolerance_us", config.plan.impact.fly_time_tolerance);
  readMillisecondsAsSeconds(
    planning, "high_speed_delay_ms", config.plan.impact.high_speed_delay_time);
  readMillisecondsAsSeconds(
    planning, "low_speed_delay_ms", config.plan.impact.low_speed_delay_time);
  readValue(
    planning, "decision_speed_rad_s", config.plan.impact.decision_speed);
  readDegrees(planning, "yaw_offset_deg", config.plan.impact.yaw_offset);
  readDegrees(planning, "pitch_offset_deg", config.plan.impact.pitch_offset);
  readValue(
    planning,
    "fallback_bullet_speed_mps",
    config.plan.impact.fallback_bullet_speed);
  readValue(
    planning,
    "min_valid_bullet_speed_mps",
    config.plan.impact.min_valid_bullet_speed);
  readDegrees(
    planning, "coming_angle_deg", config.plan.selector.coming_angle);
  readDegrees(
    planning, "leaving_angle_deg", config.plan.selector.leaving_angle);
  readDegrees(
    planning, "outpost_coming_angle_deg", config.plan.selector.outpost_coming_angle);
  readDegrees(
    planning, "outpost_leaving_angle_deg", config.plan.selector.outpost_leaving_angle);
  // 不写这一项就表示还没在实车上标定：Planner 会把计划降级成 TrackOnly，
  // 云台照常跟随但不允许开火。写了才算标定完成。
  if (planning && planning["send_to_control_ms"]) {
    config.plan.impact.send_to_control =
      planning["send_to_control_ms"].as<double>() * 1e-3;
  }

  const YAML::Node fire = root["fire"];
  readValue(fire, "shoot_enable", config.fire.shoot_enable);
  readValue(fire, "armor_width_small_m", config.fire.armor_width_small);
  readValue(fire, "armor_width_big_m", config.fire.armor_width_big);
  readValue(fire, "armor_height_m", config.fire.armor_height);
  readValue(fire, "hit_margin_ratio", config.fire.hit_margin_ratio);
  readDegrees(
    fire, "min_yaw_tolerance_deg", config.fire.min_yaw_tolerance);
  readDegrees(
    fire, "min_pitch_tolerance_deg", config.fire.min_pitch_tolerance);

  const YAML::Node runtime = root["runtime"];
  readDegrees(
    runtime,
    "command_jump_deg",
    config.runtime.command_jump_threshold);

  const YAML::Node ballistic = root["ballistic"];
  readValue(ballistic, "gravity", config.plan.ballistic.gravity);
  readValue(
    ballistic, "drag_coefficient", config.plan.ballistic.drag_coefficient);
  readDegrees(ballistic, "max_pitch_deg", config.plan.ballistic.max_pitch);

  const YAML::Node debug = root["debug"];
  readValue(debug, "overlay", config.debug.overlay);
  readValue(debug, "overlay_every", config.debug.overlay_every);
  readValue(debug, "plot", config.debug.plot);
  readValue(debug, "plot_host", config.debug.plot_host);
  readValue(debug, "plot_port", config.debug.plot_port);

  normalize(config);

  L6Telemetry::logInfo(
    "auto-aim config loaded", path,
    std::string{L2Perception::inferenceBackendName(config.inference_backend)},
    config.model_path.string(), config.inference_device);
  return config;
}

}  // namespace runtime
