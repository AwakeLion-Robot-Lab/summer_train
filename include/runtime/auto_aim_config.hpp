#pragma once

#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"

#include <filesystem>
#include <string>

namespace runtime {

struct AutoAimConfig {
  std::filesystem::path model_path{"model/armor_model/yolov5.xml"};
  std::string inference_device{"CPU"};
  L2Perception::InferenceBackendKind inference_backend{
    L2Perception::InferenceBackendKind::OpenVino};

  L3Estimation::ArmorConfig armor;
  L3Estimation::TrackerConfig tracker;
  L4Planning::PlanConfig plan;
  L5Control::FireConfig fire;

  [[nodiscard]] bool fireReady(
    const L3Estimation::AimCalibration& calibration) const noexcept
  {
    return fire.shoot_enable && calibration.fireReady() && plan.fireDelayReady() &&
           fire.parametersReady();
  }
};

// 只加载与推理后端有关的可选配置；L3/L4/L5 继续沿用现有代码默认值。
[[nodiscard]] AutoAimConfig loadAutoAimConfig(const std::string& path);

}  // namespace runtime
