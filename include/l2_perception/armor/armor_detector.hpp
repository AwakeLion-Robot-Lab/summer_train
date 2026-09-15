#pragma once

#include "l2_perception/armor/light_decoder.hpp"
#include "l2_perception/armor/light_matcher.hpp"
#include "l2_perception/armor/number_classifier.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"

#include <memory>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 一对配出的灯条及其数字分类结果，包含被拒绝的。只用于离线统计和叠加显示。
struct ArmorCandidate
{
  LightPair pair;
  NumberResult number;
};

// L2 装甲检测编排层：图像预处理 → 灯条关键点模型 → 灯条两两配对 → 数字分类。
// 通过分类的配对成为装甲板（角点就是灯条端点、类别来自数字），全部灯条另外
// 交给 L3 做独立 UVL 观测。它不拥有 PnP、跟踪或开火策略；这些工作在 L3/L4/L5。
class ArmorDetector
{
public:
  // 默认构造表示“未配置模型”，detect() 会安全返回空结果，便于 runtime 先启动相机和串口。
  ArmorDetector() = default;
  // backend 必须已加载灯条关键点模型，classifier 必须已 load()。灯条模型的输出
  // 形状在这里核对一次，不符时抛异常——启动阶段报错，而不是每帧解出垃圾。
  ArmorDetector(
    std::unique_ptr<IInferenceBackend> backend, NumberClassifier classifier,
    LightDecoderConfig decoder_config = {}, LightMatcherConfig matcher_config = {},
    ImagePreprocessConfig preprocess_config = {});

  bool ready() const noexcept;
  // 一帧同步检测。Backend/分类器抛出的异常会被转换为日志和空结果，避免中断主循环。
  [[nodiscard]] std::vector<Armor> detect(const cv::Mat& image) const;

  // IESKF 使用的完整帧入口。
  //   net_roi   网络只跑在这块区域上，缺省为整图；
  //   light_roi 有值时 lights 只保留中心落在其中的灯条，无值时 lights 为空——
  //             L3 只在跟踪中才用独立灯条，ROI 外的灯条多半属于别的车；
  //   color     只配对、只输出该颜色的灯条，Unknown 表示红蓝都要。
  [[nodiscard]] ArmorFrame detectFrame(
    const cv::Mat& image, const std::optional<cv::Rect>& light_roi = std::nullopt,
    const std::optional<cv::Rect>& net_roi = std::nullopt,
    ArmorColor color = ArmorColor::Unknown) const;

  // 网络输入的宽高比（宽 / 高）。L3 的 netFocusRoi 用它把 ROI 修成同一比例，
  // 减少 letterbox padding。后端不可用时返回 1.0。
  [[nodiscard]] double net_aspect_ratio() const noexcept;

  // 最近一帧的调试快照：颜色过滤后的全部灯条（不受 light_roi 限制）和全部配对。
  const std::vector<Light>& lastLights() const noexcept { return last_lights_; }
  const std::vector<ArmorCandidate>& lastCandidates() const noexcept
  {
    return last_candidates_;
  }
  const NumberClassifier& classifier() const noexcept { return classifier_; }

private:
  std::unique_ptr<IInferenceBackend> backend_;
  NumberClassifier classifier_;
  LightDecoder decoder_{};
  LightMatcherConfig matcher_config_{};
  ImagePreprocessConfig preprocess_config_{};
  // detect() 对外是 const 的只读操作，快照只作调试用，与 EskfTracker::observations() 同理。
  mutable std::vector<Light> last_lights_;
  mutable std::vector<ArmorCandidate> last_candidates_;
};

}  // namespace L2Perception
