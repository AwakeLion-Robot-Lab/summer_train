#pragma once

#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/filter_est/config.hpp"
#include "l3_estimation/gtsam_est/config.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"

#include <filesystem>
#include <string>

namespace runtime {

// 一次自瞄运行需要的全部参数，各层各持一段。默认值写在各自的结构体里，
// YAML 只覆盖需要调的项。
struct AutoAimConfig {
  std::filesystem::path model_path{"model/armor_model/yolov5.xml"};
  std::string inference_device{"CPU"};
  L2Perception::InferenceBackendKind inference_backend{
    L2Perception::InferenceBackendKind::OpenVino};

  // 整车状态估计器的后端。gtsam 需要 xmake f --use_gtsam=y，否则构造 Tracker
  // 时抛异常——选错后端不静默回退，见 l3_estimation/tracker.hpp。
  L3Estimation::EstimatorBackend estimator{L3Estimation::EstimatorBackend::Filter};

  L3Estimation::ArmorConfig armor;
  L3Estimation::TrackerConfig tracker;
  L3Estimation::TargetConfig target;
  L3Estimation::FilterEst::TargetConfig filter;
  L3Estimation::GtsamEst::Config gtsam;
  L4Planning::PlanConfig plan;
  L5Control::FireConfig fire;
};

// 从 YAML 读取参数。文件不存在或某个键缺失时保留结构体默认值，只在文件本身
// 解析失败时抛出——参数缺失不该让整条链路起不来。
[[nodiscard]] AutoAimConfig loadAutoAimConfig(const std::string& path);

}  // namespace runtime
