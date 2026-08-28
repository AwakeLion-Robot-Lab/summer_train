#pragma once

#include "l2_perception/armor/armor_decoder.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l4_planning/armor/types.hpp"
#include "l5_control/fire_decision.hpp"

#include <filesystem>
#include <numbers>
#include <string>

namespace runtime {

// 调试叠加层。实机默认全关：cv::imshow 在 1440x1080 上要几毫秒，直接计入
// image_to_plan；而且比赛用的 NUC 上根本没有显示器，无条件 namedWindow 会抛。
struct DebugConfig {
  bool overlay{false};
  // 每 N 帧画一次。画面只是用来目视对齐，不必每帧都画。
  int overlay_every{1};
};

// 只负责 runtime 胶水层的相邻命令检查，不重复 L3/L4/L5 的业务参数。
struct RuntimeSafetyConfig {
  double command_jump_threshold{10.0 * std::numbers::pi / 180.0};
};

struct AutoAimConfig {
  std::filesystem::path model_path{"model/armor_model/yolov5.xml"};
  std::string inference_device{"CPU"};
  L2Perception::InferenceBackendKind inference_backend{
    L2Perception::InferenceBackendKind::OpenVino};

  // 后端调度和模型格式参数。字段含义见 InferenceModelConfig，那里是唯一的
  // 权威定义，这里只负责从 YAML 搬运，不复制默认值以外的语义。
  //
  // model_path / device / backend 单列在上面是因为 runtime 自己也要用它们
  // 打日志和选后端；其余的只在构造 Backend 时透传，所以整个结构体直接放这。
  L2Perception::InferenceModelConfig inference;

  // 模型输出契约。和 model_path 是一对：换模型必须同时换契约，否则解码出的
  // 是垃圾角点而不是报错。默认是 SP YOLOV5 的 [1, 25200, 22]。
  L2Perception::ArmorDecoderConfig decoder;

  L3Estimation::ArmorConfig armor;
  L3Estimation::TrackerConfig tracker;
  L3Estimation::TargetConfig target;
  L4Planning::ArmorPlanConfig plan;
  L5Control::FireConfig fire;
  RuntimeSafetyConfig runtime;
  DebugConfig debug;
};

// 缺失字段保留各层的安全默认值；特别是 shoot_enable 默认为 false。
[[nodiscard]] AutoAimConfig loadAutoAimConfig(const std::string& path);

}  // namespace runtime
