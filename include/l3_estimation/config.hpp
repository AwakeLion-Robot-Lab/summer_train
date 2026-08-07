#pragma once

#include "l3_estimation/ekf_tracker.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/types.hpp"
#include "l3_estimation/yaw_optimizer.hpp"

#include <filesystem>

namespace L3Estimation {

// 单个目标模型的固定参数：pitch、初始半径、过程噪声和过期时间。
struct TargetModelParameters {
  double pitch_rad = 0.0;
  double initial_radius_m = 0.20;
  double linear_acceleration_variance = 100.0;
  double angular_acceleration_variance = 400.0;
  std::chrono::milliseconds expiration_timeout{500};
};

// 装甲尺寸与两个模型的独立参数。
struct ArmorModelConfig {
  ArmorDimensions dimensions{};
  TargetModelParameters four_armor_vehicle{};
  TargetModelParameters three_armor_outpost{
    .pitch_rad = -0.2617993877991494,
    .initial_radius_m = 0.2765,
    .linear_acceleration_variance = 10.0,
    .angular_acceleration_variance = 0.1,
    .expiration_timeout = std::chrono::milliseconds{2500}};

  // 按模型取参数。
  [[nodiscard]] const TargetModelParameters& parameters(
    TargetModel model) const noexcept
  {
    return model == TargetModel::ThreeArmorOutpost
             ? three_armor_outpost
             : four_armor_vehicle;
  }
};

// L3 全量配置：图像尺寸校验、装甲/PnP/yaw 优化/tracker。
struct L3Config {
  bool require_matching_image_size = true;
  ArmorModelConfig armor{};
  PnpSolverConfig pnp{};
  YawOptimizationConfig yaw_optimization{};
  EkfTrackerConfig tracker{};

  // 把模型专属参数并入通用 tracker 配置。
  [[nodiscard]] EkfTrackerConfig trackerConfig(
    TargetModel model) const noexcept
  {
    EkfTrackerConfig result = tracker;
    const auto& model_parameters = armor.parameters(model);
    result.initial_radius = model_parameters.initial_radius_m;
    result.linear_acceleration_variance =
      model_parameters.linear_acceleration_variance;
    result.angular_acceleration_variance =
      model_parameters.angular_acceleration_variance;
    result.expiration_timeout =
      model_parameters.expiration_timeout;
    return result;
  }
};

// 全量配置合法性检查（范围、有限性、一致性）。
[[nodiscard]] bool isValidL3Config(const L3Config& config) noexcept;

// 字段缺失或单位、范围错误时直接抛出，避免带着部分默认值运行。
[[nodiscard]] L3Config loadL3Config(const std::filesystem::path& path);

}  // namespace L3Estimation
