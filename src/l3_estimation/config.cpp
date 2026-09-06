#include "l3_estimation/config.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace L3Estimation {
namespace {

[[noreturn]] void invalidConfig(
  const std::string& source,
  const std::string& reason)
{
  throw std::runtime_error(
    "invalid L3 config '" + source + "': " + reason);
}

YAML::Node requiredMap(
  const YAML::Node& parent,
  const char* key,
  const std::string& source)
{
  const YAML::Node node = parent[key];
  if (!node || !node.IsMap()) {
    invalidConfig(source, std::string{key} + " must be a YAML map");
  }
  return node;
}

template<typename Value>
Value requiredValue(
  const YAML::Node& parent,
  const char* key,
  const std::string& source)
{
  const YAML::Node node = parent[key];
  if (!node || !node.IsScalar()) {
    invalidConfig(source, std::string{key} + " must be a scalar");
  }
  try {
    return node.as<Value>();
  } catch (const YAML::Exception& error) {
    invalidConfig(
      source,
      "failed to parse " + std::string{key} + ": " + error.what());
  }
}

StateVector requiredStateVector(
  const YAML::Node& parent,
  const char* key,
  const std::string& source)
{
  const YAML::Node node = parent[key];
  if (!node || !node.IsSequence()
      || node.size() != static_cast<std::size_t>(STATE_DIM)) {
    invalidConfig(
      source,
      std::string{key} + " must contain "
        + std::to_string(STATE_DIM) + " values");
  }
  StateVector result;
  try {
    for (int index = 0; index < STATE_DIM; ++index) {
      result[index] =
        node[static_cast<std::size_t>(index)].as<double>();
    }
  } catch (const YAML::Exception& error) {
    invalidConfig(source, "invalid initial_variance: " + std::string{
      error.what()});
  }
  return result;
}

bool finitePositive(double value) noexcept
{
  return std::isfinite(value) && value > 0.0;
}

bool finiteNonNegative(double value) noexcept
{
  return std::isfinite(value) && value >= 0.0;
}

bool validModelParameters(
  const TargetModelParameters& parameters) noexcept
{
  return std::isfinite(parameters.pitch_rad)
         && std::abs(parameters.pitch_rad) <= std::numbers::pi / 2.0
         && finitePositive(parameters.initial_radius_m)
         && finiteNonNegative(
           parameters.linear_acceleration_variance)
         && finiteNonNegative(
           parameters.angular_acceleration_variance)
         && parameters.expiration_timeout.count() > 0;
}

TargetModelParameters readModelParameters(
  const YAML::Node& armor,
  const char* key,
  const std::string& source)
{
  const YAML::Node model = requiredMap(armor, key, source);
  return {
    .pitch_rad = requiredValue<double>(model, "pitch_rad", source),
    .initial_radius_m =
      requiredValue<double>(model, "initial_radius_m", source),
    .linear_acceleration_variance =
      requiredValue<double>(
        model,
        "linear_acceleration_variance",
        source),
    .angular_acceleration_variance =
      requiredValue<double>(
        model,
        "angular_acceleration_variance",
        source),
    .expiration_timeout = std::chrono::milliseconds{
      requiredValue<int>(
        model,
        "expiration_timeout_ms",
        source)}};
}

}  // namespace

bool isValidL3Config(const L3Config& config) noexcept
{
  const auto& dimensions = config.armor.dimensions;
  const auto& pnp = config.pnp;
  const auto& yaw = config.yaw_search;
  const auto& tracker = config.tracker;
  const bool tracker_base_valid =
    tracker.initial_variance.allFinite()
    && (tracker.initial_variance.array() > 0.0).all()
    && finiteNonNegative(tracker.geometry_random_walk_variance)
    && finitePositive(tracker.position_standard_deviation_base_m)
    && finiteNonNegative(
      tracker.position_standard_deviation_quadratic)
    && finitePositive(tracker.armor_yaw_standard_deviation_rad)
    && tracker.confirmation_hits > 0
    && tracker.max_predict_interval.count() > 0
    && finitePositive(tracker.association_position_gate)
    && finitePositive(tracker.association_yaw_gate)
    && tracker.association_yaw_gate <= std::numbers::pi
    && finiteNonNegative(tracker.association_position_weight)
    && finiteNonNegative(tracker.association_yaw_weight)
    && finitePositive(tracker.nis_reference_threshold)
    && finitePositive(tracker.min_radius)
    && finitePositive(tracker.max_radius)
    && tracker.max_radius > tracker.min_radius
    && finiteNonNegative(tracker.max_abs_height_offset);

  return finitePositive(dimensions.small_width)
         && finitePositive(dimensions.large_width)
         && finitePositive(dimensions.height)
         && validModelParameters(config.armor.four_armor_vehicle)
         && validModelParameters(config.armor.three_armor_outpost)
         && config.armor.four_armor_vehicle.initial_radius_m
              >= tracker.min_radius
         && config.armor.four_armor_vehicle.initial_radius_m
              <= tracker.max_radius
         && config.armor.three_armor_outpost.initial_radius_m
              >= tracker.min_radius
         && config.armor.three_armor_outpost.initial_radius_m
              <= tracker.max_radius
         && finitePositive(pnp.minimum_corner_area_px)
         && finiteNonNegative(pnp.minimum_distance_m)
         && finitePositive(pnp.maximum_distance_m)
         && pnp.maximum_distance_m > pnp.minimum_distance_m
         && finitePositive(pnp.maximum_reprojection_error_px)
         && finitePositive(yaw.search_half_range_rad)
         && yaw.search_half_range_rad <= std::numbers::pi
         && finitePositive(yaw.search_step_rad)
         && yaw.search_step_rad <= yaw.search_half_range_rad
         && tracker_base_valid;
}

L3Config loadL3Config(const std::filesystem::path& path)
{
  // 先检查文件根节点和版本，避免静默读取不兼容配置。
  const std::string source = path.string();
  YAML::Node root;
  try {
    root = YAML::LoadFile(source);
  } catch (const YAML::Exception& error) {
    invalidConfig(source, error.what());
  }
  if (!root || !root.IsMap()) {
    invalidConfig(source, "root must be a YAML map");
  }
  const int version = requiredValue<int>(root, "version", source);
  if (version != 1) {
    invalidConfig(
      source,
      "unsupported version " + std::to_string(version));
  }

  L3Config config;
  // 第 0 轮接口配置：图像尺寸检查和两类目标的几何参数。
  config.require_matching_image_size =
    requiredValue<bool>(
      root,
      "require_matching_image_size",
      source);

  const YAML::Node armor = requiredMap(root, "armor", source);
  config.armor.dimensions.small_width =
    requiredValue<double>(armor, "small_width_m", source);
  config.armor.dimensions.large_width =
    requiredValue<double>(armor, "large_width_m", source);
  config.armor.dimensions.height =
    requiredValue<double>(armor, "height_m", source);
  config.armor.four_armor_vehicle =
    readModelParameters(
      armor,
      "four_armor_vehicle",
      source);
  config.armor.three_armor_outpost =
    readModelParameters(
      armor,
      "three_armor_outpost",
      source);

  // 第一版 PnP：单个 IPPE 解及其输入、距离和像素误差门限。
  const YAML::Node pnp = requiredMap(root, "pnp", source);
  if (requiredValue<std::string>(pnp, "method", source) != "IPPE") {
    invalidConfig(source, "pnp.method must be IPPE");
  }
  config.pnp.minimum_corner_area_px =
    requiredValue<double>(pnp, "minimum_corner_area_px", source);
  config.pnp.minimum_distance_m =
    requiredValue<double>(pnp, "minimum_distance_m", source);
  config.pnp.maximum_distance_m =
    requiredValue<double>(pnp, "maximum_distance_m", source);
  config.pnp.maximum_reprojection_error_px =
    requiredValue<double>(
      pnp,
      "maximum_reprojection_error_px",
      source);

  // 第一版 yaw：固定位置和 pitch，只配置遍历范围与步长。
  const YAML::Node yaw = requiredMap(root, "yaw_search", source);
  config.yaw_search.enabled =
    requiredValue<bool>(yaw, "enabled", source);
  config.yaw_search.search_half_range_rad =
    requiredValue<double>(yaw, "half_range_rad", source);
  config.yaw_search.search_step_rad =
    requiredValue<double>(yaw, "step_rad", source);

  // 第一版 EKF：读取观测噪声、简单关联和生命周期参数。
  const YAML::Node noise =
    requiredMap(root, "observation_noise", source);
  config.tracker.position_standard_deviation_base_m =
    requiredValue<double>(
      noise,
      "position_standard_deviation_base_m",
      source);
  config.tracker.position_standard_deviation_quadratic =
    requiredValue<double>(
      noise,
      "position_standard_deviation_quadratic",
      source);
  config.tracker.armor_yaw_standard_deviation_rad =
    requiredValue<double>(
      noise,
      "armor_yaw_standard_deviation_rad",
      source);

  const YAML::Node association =
    requiredMap(root, "association", source);
  config.tracker.association_position_gate =
    requiredValue<double>(
      association,
      "position_gate_m",
      source);
  config.tracker.association_yaw_gate =
    requiredValue<double>(
      association,
      "yaw_gate_rad",
      source);
  config.tracker.association_position_weight =
    requiredValue<double>(
      association,
      "position_weight",
      source);
  config.tracker.association_yaw_weight =
    requiredValue<double>(
      association,
      "yaw_weight",
      source);
  config.tracker.nis_reference_threshold =
    requiredValue<double>(
      association,
      "nis_reference_threshold",
      source);

  const YAML::Node lifecycle =
    requiredMap(root, "lifecycle", source);
  config.tracker.confirmation_hits =
    requiredValue<int>(
      lifecycle,
      "confirmation_hits",
      source);
  config.tracker.max_predict_interval = std::chrono::milliseconds{
    requiredValue<int>(
      lifecycle,
      "max_predict_interval_ms",
      source)};

  // 11 维状态的初始方差和车辆几何安全范围。
  const YAML::Node tracker = requiredMap(root, "tracker", source);
  config.tracker.initial_variance =
    requiredStateVector(
      tracker,
      "initial_variance",
      source);
  config.tracker.geometry_random_walk_variance =
    requiredValue<double>(
      tracker,
      "geometry_random_walk_variance",
      source);
  config.tracker.min_radius =
    requiredValue<double>(
      tracker,
      "minimum_radius_m",
      source);
  config.tracker.max_radius =
    requiredValue<double>(
      tracker,
      "maximum_radius_m",
      source);
  config.tracker.max_abs_height_offset =
    requiredValue<double>(
      tracker,
      "maximum_absolute_height_offset_m",
      source);

  if (!isValidL3Config(config)) {
    invalidConfig(
      source,
      "one or more values are outside valid ranges");
  }
  return config;
}

}  // namespace L3Estimation
