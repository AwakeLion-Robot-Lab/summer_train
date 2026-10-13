#pragma once

#include "l2_perception/armor/armor_decoder.hpp"
#include "l2_perception/armor/light_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l3_estimation/armor/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace runtime {

/**
 * @brief 调试旁路的开关
 *
 * 实机默认全关：cv::imshow 在 1440x1080 上要几毫秒，直接计入 image_to_plan；
 * 而且比赛用的 NUC 上根本没有显示器，无条件 namedWindow 会抛。
 */
struct DebugConfig {
  /// 是否开叠加层窗口
  bool overlay{false};

  /// 每 N 帧画一次。画面只是用来目视对齐，不必每帧都画。
  int overlay_every{1};

  /// PlotJuggler 遥测。与叠加层分开开关：上车查云台抖动时 NUC 没显示器，要的是
  /// 曲线；UDP 无连接，没人接收也不阻塞。
  bool plot{false};

  /// 遥测目标 IP
  std::string plot_host{"127.0.0.1"};

  /// 遥测目标端口
  int plot_port{9870};

  /// 无视下位机上报的 WorkMode，强制按指定模式跑。空串表示不覆盖。
  /// 只用于电控还没接好模式切换、但视觉侧要先把链路跑通的场合。
  /// 这是**调试用的旁路**：正常比赛必须留空，由下位机决定何时进自瞄。
  /// 注意它不解除开火闸门，shoot_enable 仍然独立生效。
  std::string force_work_mode{};
};

/**
 * @brief inference.decoder 里显式写出的阈值
 *
 * layout 为 auto 时预设要等模型加载、按输出形状认出来之后才定，阈值得在那之后
 * 覆盖上去，所以单独留一份。没写的项为空。
 */
struct DecoderThresholds {
  /// 见 ArmorDecoderConfig::confidence_threshold
  std::optional<float> confidence_threshold;

  /// 见 ArmorDecoderConfig::minimum_confidence
  std::optional<float> minimum_confidence;

  /// 见 ArmorDecoderConfig::nms_iou_threshold
  std::optional<float> nms_iou_threshold;

  /// 见 ArmorDecoderConfig::nms_score_threshold
  std::optional<float> nms_score_threshold;
};

/**
 * @brief config/auto_aim.yaml 的全部内容
 *
 * YAML 里角度写度、时间写毫秒，加载时已换成 rad 与秒。
 */
struct AutoAimConfig {
  /// 整板检测模型，输出契约由 decoder 的 layout 指定。
  std::filesystem::path model_path{"model/armor_model/yolov5.xml"};

  /// 推理设备
  std::string inference_device{"CPU"};

  /// 推理后端
  L2Perception::InferenceBackendKind inference_backend{
    L2Perception::InferenceBackendKind::OpenVino};

  /// 后端调度和模型格式参数。字段含义见 InferenceModelConfig，那里是唯一的
  /// 权威定义，这里只负责从 YAML 搬运，不复制默认值以外的语义。
  ///
  /// model_path / device / backend 单列在上面是因为 runtime 自己也要用它们
  /// 打日志和选后端；其余的只在构造 Backend 时透传，所以整个结构体直接放这。
  L2Perception::InferenceModelConfig inference;

  /// 整板模型的输出契约与筛选阈值（inference.decoder），输出形状在构造
  /// ArmorDetector 时核对。
  L2Perception::ArmorDecoderConfig decoder;

  /// layout: auto 时为 true：decoder 里的契约作废，makeDetector 按模型输出形状
  /// 选预设，再套上 decoder_thresholds。
  bool auto_layout{false};

  /// YAML 里显式写出的解码阈值
  DecoderThresholds decoder_thresholds;

  /// 侧边灯条：剖面搜索的门限与判色阈值。这一路不走网络。
  L2Perception::LightFinderConfig light_finder;

  /// 装甲板几何尺寸
  L3Estimation::ArmorConfig armor;

  /// 跟踪器状态机与 ROI 参数
  L3Estimation::EskfTrackerConfig ieskf_tracker;

  /// 整车滤波器参数
  L3Estimation::EskfTargetConfig ieskf_target;

  /// 规划器参数
  L4Planning::PlanConfig plan;

  /// 开火判定参数
  L5Control::FireConfig fire;

  /// 调试旁路开关
  DebugConfig debug;
};

/**
 * @brief 读取自瞄配置
 *
 * 缺失字段保留各层的安全默认值；特别是 shoot_enable 默认为 false。
 *
 * @param path YAML 路径（如 config/auto_aim.yaml）
 * @return 自瞄配置；文件不存在时全部取默认值，字段取值非法时抛异常
 */
[[nodiscard]] AutoAimConfig loadConfig(const std::string& path);

/**
 * @brief 把写了的阈值盖到 decoder 上
 * @param thresholds YAML 里显式写出的阈值
 * @param decoder 要覆盖的解码配置；越界的值（不在 [0, 1]）丢掉、保留预设的
 */
void applyThresholds(
  const DecoderThresholds& thresholds, L2Perception::ArmorDecoderConfig& decoder);

}  // namespace runtime
