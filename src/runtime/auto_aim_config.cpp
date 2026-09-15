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

void readVector3(
  const YAML::Node& section,
  const char* key,
  Eigen::Vector3d& value)
{
  if (!section || !section[key]) {
    return;
  }
  try {
    const YAML::Node vector = section[key];
    if (!vector.IsSequence() || vector.size() != 3) {
      throw std::runtime_error("expected a three-element sequence");
    }
    Eigen::Vector3d parsed;
    parsed << vector[0].as<double>(), vector[1].as<double>(), vector[2].as<double>();
    value = parsed;
  } catch (const std::exception& error) {
    L6Telemetry::logWarn("auto-aim config invalid field", key, error.what());
  }
}

bool positiveFinite(double value) noexcept
{
  return std::isfinite(value) && value > 0.0;
}

bool nonNegativeFinite(double value) noexcept
{
  return std::isfinite(value) && value >= 0.0;
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
  if (!(config.refiner.independent_light_binary_threshold > 0.0 &&
        config.refiner.independent_light_binary_threshold < 255.0)) {
    config.refiner.independent_light_binary_threshold =
      refiner_defaults.independent_light_binary_threshold;
  }
  if (!nonNegativeFinite(
        config.refiner.independent_light_threshold_tolerance)) {
    config.refiner.independent_light_threshold_tolerance =
      refiner_defaults.independent_light_threshold_tolerance;
  }
  if (!nonNegativeFinite(
        config.refiner.independent_light_color_diff_threshold)) {
    config.refiner.independent_light_color_diff_threshold =
      refiner_defaults.independent_light_color_diff_threshold;
  }
  if (!positiveFinite(
        config.refiner.independent_light_min_contour_area_px)) {
    config.refiner.independent_light_min_contour_area_px =
      refiner_defaults.independent_light_min_contour_area_px;
  }
  if (!(config.refiner.independent_light_min_fill_ratio > 0.0 &&
        config.refiner.independent_light_min_fill_ratio <= 1.0)) {
    config.refiner.independent_light_min_fill_ratio =
      refiner_defaults.independent_light_min_fill_ratio;
  }
  if (!positiveFinite(config.refiner.independent_light_min_length_px)) {
    config.refiner.independent_light_min_length_px =
      refiner_defaults.independent_light_min_length_px;
  }
  if (!(config.refiner.independent_light_min_width_length_ratio > 0.0F &&
        config.refiner.independent_light_min_width_length_ratio <
          config.refiner.independent_light_max_width_length_ratio)) {
    config.refiner.independent_light_min_width_length_ratio =
      refiner_defaults.independent_light_min_width_length_ratio;
    config.refiner.independent_light_max_width_length_ratio =
      refiner_defaults.independent_light_max_width_length_ratio;
  }
  if (!(config.refiner.independent_light_max_tilt_angle_deg > 0.0F &&
        config.refiner.independent_light_max_tilt_angle_deg <= 90.0F)) {
    config.refiner.independent_light_max_tilt_angle_deg =
      refiner_defaults.independent_light_max_tilt_angle_deg;
  }
  const auto hueRangeValid = [](int low, int high) {
    return low >= 0 && low <= high && high <= 180;
  };
  const auto channelMinimumValid = [](int value) {
    return value >= 0 && value <= 255;
  };
  if (!hueRangeValid(
        config.refiner.independent_light_red_h_min_low,
        config.refiner.independent_light_red_h_max_low) ||
      !hueRangeValid(
        config.refiner.independent_light_red_h_min_high,
        config.refiner.independent_light_red_h_max_high) ||
      !channelMinimumValid(config.refiner.independent_light_red_s_min) ||
      !channelMinimumValid(config.refiner.independent_light_red_v_min)) {
    config.refiner.independent_light_red_h_min_low =
      refiner_defaults.independent_light_red_h_min_low;
    config.refiner.independent_light_red_h_max_low =
      refiner_defaults.independent_light_red_h_max_low;
    config.refiner.independent_light_red_h_min_high =
      refiner_defaults.independent_light_red_h_min_high;
    config.refiner.independent_light_red_h_max_high =
      refiner_defaults.independent_light_red_h_max_high;
    config.refiner.independent_light_red_s_min =
      refiner_defaults.independent_light_red_s_min;
    config.refiner.independent_light_red_v_min =
      refiner_defaults.independent_light_red_v_min;
  }
  if (!hueRangeValid(
        config.refiner.independent_light_blue_h_min,
        config.refiner.independent_light_blue_h_max) ||
      !channelMinimumValid(config.refiner.independent_light_blue_s_min) ||
      !channelMinimumValid(config.refiner.independent_light_blue_v_min)) {
    config.refiner.independent_light_blue_h_min =
      refiner_defaults.independent_light_blue_h_min;
    config.refiner.independent_light_blue_h_max =
      refiner_defaults.independent_light_blue_h_max;
    config.refiner.independent_light_blue_s_min =
      refiner_defaults.independent_light_blue_s_min;
    config.refiner.independent_light_blue_v_min =
      refiner_defaults.independent_light_blue_v_min;
  }
  config.refiner.independent_light_morphology_width = std::clamp(
    config.refiner.independent_light_morphology_width, 1, 31);
  config.refiner.independent_light_morphology_height = std::clamp(
    config.refiner.independent_light_morphology_height, 1, 31);

  // awakening IESKF：状态机按真实时间计，过程噪声在车体系表达，UVL 的观测
  // 噪声是 sigma 而非方差。
  const L3Estimation::EskfTrackerConfig ieskf_tracker_defaults;
  config.ieskf_tracker.tracking_thres =
    std::max(config.ieskf_tracker.tracking_thres, 1);
  if (!positiveFinite(config.ieskf_tracker.lost_time_thres)) {
    config.ieskf_tracker.lost_time_thres = ieskf_tracker_defaults.lost_time_thres;
  }
  if (!positiveFinite(config.ieskf_tracker.lost_time_thres_outpost)) {
    config.ieskf_tracker.lost_time_thres_outpost =
      ieskf_tracker_defaults.lost_time_thres_outpost;
  }
  config.ieskf_tracker.lost_time_thres_outpost = std::max(
    config.ieskf_tracker.lost_time_thres_outpost,
    config.ieskf_tracker.lost_time_thres);

  const L3Estimation::EskfTargetConfig ieskf_target_defaults;
  config.ieskf_target.iteration_num = std::max(config.ieskf_target.iteration_num, 1);
  for (int axis = 0; axis < 3; ++axis) {
    if (!positiveFinite(config.ieskf_target.noise.body_acceleration[axis])) {
      config.ieskf_target.noise.body_acceleration[axis] =
        ieskf_target_defaults.noise.body_acceleration[axis];
    }
    if (!positiveFinite(config.ieskf_target.noise.outpost_body_acceleration[axis])) {
      config.ieskf_target.noise.outpost_body_acceleration[axis] =
        ieskf_target_defaults.noise.outpost_body_acceleration[axis];
    }
  }
  for (const auto& [value, fallback] : {
         std::pair{&config.ieskf_target.noise.yaw_acceleration,
                   ieskf_target_defaults.noise.yaw_acceleration},
         std::pair{&config.ieskf_target.noise.outpost_yaw_acceleration,
                   ieskf_target_defaults.noise.outpost_yaw_acceleration},
         std::pair{&config.ieskf_target.noise.radius,
                   ieskf_target_defaults.noise.radius},
         std::pair{&config.ieskf_target.noise.height,
                   ieskf_target_defaults.noise.height},
         std::pair{&config.ieskf_target.noise.outpost_height,
                   ieskf_target_defaults.noise.outpost_height},
         std::pair{&config.ieskf_target.noise.roll_pitch,
                   ieskf_target_defaults.noise.roll_pitch},
         std::pair{&config.ieskf_target.sigma_pixel_by_length,
                   ieskf_target_defaults.sigma_pixel_by_length},
         std::pair{&config.ieskf_target.sigma_length_by_length,
                   ieskf_target_defaults.sigma_length_by_length},
         std::pair{&config.ieskf_target.sigma_angle,
                   ieskf_target_defaults.sigma_angle},
         std::pair{&config.ieskf_target.isolated_light_sigma_scale,
                   ieskf_target_defaults.isolated_light_sigma_scale},
         std::pair{&config.ieskf_target.armor_lights_depth_diff_sigma,
                   ieskf_target_defaults.armor_lights_depth_diff_sigma},
         std::pair{&config.ieskf_target.light_match_length_ratio_gate,
                   ieskf_target_defaults.light_match_length_ratio_gate},
         std::pair{&config.ieskf_target.light_match_angle_gate,
                   ieskf_target_defaults.light_match_angle_gate},
         std::pair{&config.ieskf_target.light_match_pos_gate_by_length_ratio,
                   ieskf_target_defaults.light_match_pos_gate_by_length_ratio},
         std::pair{&config.ieskf_target.match_gate,
                   ieskf_target_defaults.match_gate},
         std::pair{&config.ieskf_target.match_gate_not_all_init,
                   ieskf_target_defaults.match_gate_not_all_init},
         std::pair{&config.ieskf_target.initial_radius,
                   ieskf_target_defaults.initial_radius},
         std::pair{&config.ieskf_target.initial_radius_outpost,
                   ieskf_target_defaults.initial_radius_outpost},
         std::pair{&config.ieskf_target.initial_radius_base,
                   ieskf_target_defaults.initial_radius_base}}) {
    if (!positiveFinite(*value)) {
      *value = fallback;
    }
  }
  for (const auto& [value, fallback] : {
         std::pair{&config.ieskf_target.weight_center_error,
                   ieskf_target_defaults.weight_center_error},
         std::pair{&config.ieskf_target.weight_angle_error,
                   ieskf_target_defaults.weight_angle_error},
         std::pair{&config.ieskf_target.weight_side_length_error,
                   ieskf_target_defaults.weight_side_length_error}}) {
    if (!nonNegativeFinite(*value)) {
      *value = fallback;
    }
  }
  config.ieskf_target.match_gate_not_all_init = std::max(
    config.ieskf_target.match_gate_not_all_init,
    config.ieskf_target.match_gate);
  // UVL 和 PnP 必须引用同一套物理板尺寸。
  config.ieskf_target.armor = config.armor;

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
  readValue(
    refiner, "pca_corner_correction", config.refiner.pca_corner_correction);
  readValue(
    refiner, "independent_light_enable",
    config.refiner.independent_light_enable);
  readValue(
    refiner, "independent_light_binary_threshold",
    config.refiner.independent_light_binary_threshold);
  readValue(
    refiner, "independent_light_threshold_tolerance",
    config.refiner.independent_light_threshold_tolerance);
  readValue(
    refiner, "independent_light_color_diff_threshold",
    config.refiner.independent_light_color_diff_threshold);
  readValue(
    refiner, "independent_light_min_contour_area_px",
    config.refiner.independent_light_min_contour_area_px);
  readValue(
    refiner, "independent_light_min_fill_ratio",
    config.refiner.independent_light_min_fill_ratio);
  readValue(
    refiner, "independent_light_min_length_px",
    config.refiner.independent_light_min_length_px);
  readValue(
    refiner, "independent_light_min_width_length_ratio",
    config.refiner.independent_light_min_width_length_ratio);
  readValue(
    refiner, "independent_light_max_width_length_ratio",
    config.refiner.independent_light_max_width_length_ratio);
  readValue(
    refiner, "independent_light_max_tilt_angle_deg",
    config.refiner.independent_light_max_tilt_angle_deg);
  readValue(
    refiner, "independent_light_red_h_min_low",
    config.refiner.independent_light_red_h_min_low);
  readValue(
    refiner, "independent_light_red_h_max_low",
    config.refiner.independent_light_red_h_max_low);
  readValue(
    refiner, "independent_light_red_h_min_high",
    config.refiner.independent_light_red_h_min_high);
  readValue(
    refiner, "independent_light_red_h_max_high",
    config.refiner.independent_light_red_h_max_high);
  readValue(
    refiner, "independent_light_red_s_min",
    config.refiner.independent_light_red_s_min);
  readValue(
    refiner, "independent_light_red_v_min",
    config.refiner.independent_light_red_v_min);
  readValue(
    refiner, "independent_light_blue_h_min",
    config.refiner.independent_light_blue_h_min);
  readValue(
    refiner, "independent_light_blue_h_max",
    config.refiner.independent_light_blue_h_max);
  readValue(
    refiner, "independent_light_blue_s_min",
    config.refiner.independent_light_blue_s_min);
  readValue(
    refiner, "independent_light_blue_v_min",
    config.refiner.independent_light_blue_v_min);
  readValue(
    refiner, "independent_light_use_morphology",
    config.refiner.independent_light_use_morphology);
  readValue(
    refiner, "independent_light_morphology_width",
    config.refiner.independent_light_morphology_width);
  readValue(
    refiner, "independent_light_morphology_height",
    config.refiner.independent_light_morphology_height);

  const YAML::Node ieskf = root["ieskf"];
  readValue(ieskf, "tracking_thres", config.ieskf_tracker.tracking_thres);
  readValue(ieskf, "lost_time_thres", config.ieskf_tracker.lost_time_thres);
  readValue(
    ieskf, "lost_time_thres_outpost", config.ieskf_tracker.lost_time_thres_outpost);
  readValue(ieskf, "iteration_num", config.ieskf_target.iteration_num);
  readVector3(
    ieskf, "body_acceleration", config.ieskf_target.noise.body_acceleration);
  readValue(
    ieskf, "yaw_acceleration", config.ieskf_target.noise.yaw_acceleration);
  readVector3(
    ieskf, "outpost_body_acceleration",
    config.ieskf_target.noise.outpost_body_acceleration);
  readValue(
    ieskf, "outpost_yaw_acceleration",
    config.ieskf_target.noise.outpost_yaw_acceleration);
  readValue(ieskf, "radius_noise", config.ieskf_target.noise.radius);
  readValue(ieskf, "height_noise", config.ieskf_target.noise.height);
  readValue(
    ieskf, "outpost_height_noise", config.ieskf_target.noise.outpost_height);
  readValue(ieskf, "roll_pitch_noise", config.ieskf_target.noise.roll_pitch);
  readValue(
    ieskf, "sigma_pixel_by_length", config.ieskf_target.sigma_pixel_by_length);
  readValue(
    ieskf, "sigma_length_by_length", config.ieskf_target.sigma_length_by_length);
  readValue(
    ieskf, "sigma_perp_px", config.ieskf_target.sigma_perp_px);
  readValue(
    ieskf, "isolated_light_sigma_scale",
    config.ieskf_target.isolated_light_sigma_scale);
  readValue(ieskf, "sigma_angle", config.ieskf_target.sigma_angle);
  readValue(
    ieskf, "armor_lights_depth_diff_sigma",
    config.ieskf_target.armor_lights_depth_diff_sigma);
  readValue(
    ieskf, "enable_lights_measure",
    config.ieskf_target.enable_lights_measure);
  readValue(
    ieskf, "light_match_length_ratio_gate",
    config.ieskf_target.light_match_length_ratio_gate);
  readValue(
    ieskf, "light_match_angle_gate",
    config.ieskf_target.light_match_angle_gate);
  readValue(
    ieskf, "light_match_pos_gate_by_length_ratio",
    config.ieskf_target.light_match_pos_gate_by_length_ratio);
  readValue(ieskf, "match_gate", config.ieskf_target.match_gate);
  readValue(
    ieskf, "match_gate_not_all_init", config.ieskf_target.match_gate_not_all_init);
  readValue(
    ieskf, "weight_center_error", config.ieskf_target.weight_center_error);
  readValue(
    ieskf, "weight_angle_error", config.ieskf_target.weight_angle_error);
  readValue(
    ieskf, "weight_side_length_error", config.ieskf_target.weight_side_length_error);
  readValue(ieskf, "initial_radius", config.ieskf_target.initial_radius);
  readValue(
    ieskf, "initial_radius_outpost", config.ieskf_target.initial_radius_outpost);
  readValue(
    ieskf, "initial_radius_base", config.ieskf_target.initial_radius_base);

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

  const YAML::Node debug = root["debug"];
  readValue(debug, "overlay", config.debug.overlay);
  readValue(debug, "overlay_every", config.debug.overlay_every);

  normalize(config);

  L6Telemetry::logInfo(
    "auto-aim config loaded", path,
    std::string{L2Perception::inferenceBackendName(config.inference_backend)},
    config.model_path.string(), config.inference_device);
  return config;
}

}  // namespace runtime
