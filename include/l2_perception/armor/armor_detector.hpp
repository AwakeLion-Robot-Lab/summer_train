#pragma once

#include "l2_perception/armor/light_detector.hpp"
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

// 一对配出的灯条及其数字分类结果，被数字分类拒掉的也在内。只给离线工具统计
// 和叠加显示用，主链路不读它。
struct ArmorCandidate
{
  LightPair pair;
  NumberResult number;
};

// L2 装甲检测的编排层，把四步串起来：找灯条（LightMode 决定用哪一路）→ 按
// 颜色过滤 → 两两配对 → 数字分类。通过分类的配对成为装甲板，角点就是两根
// 灯条的端点，类别来自数字；灯条本身另外交给 L3 做 UVL 观测。
//
// PnP、跟踪、开火策略都不在这里，分别属于 L3/L4/L5。
class ArmorDetector
{
public:
  // 默认构造表示“未配置模型”：ready() 为 false，detect() 返回空结果，runtime
  // 可以先把相机和串口跑起来。
  ArmorDetector() = default;
  // backend 必须已加载灯条关键点模型，classifier 必须已 load()，有一个没就绪
  // 就抛异常；随后核对一次模型输出形状，免得每帧解出垃圾端点。
  ArmorDetector(
    std::unique_ptr<IInferenceBackend> backend, NumberClassifier classifier,
    LightDecoderConfig decoder_config = {}, LightMatcherConfig matcher_config = {},
    LightFinderConfig finder_config = {}, ImagePreprocessConfig preprocess_config = {});

  bool ready() const noexcept;
  // 整图检测一帧，只要装甲板，等价于 detectFrame(image).armors。
  [[nodiscard]] std::vector<Armor> detect(const cv::Mat& image) const;

  // 完整的一帧检测，L3 的 IESKF 走这个入口。
  //   net_roi   两路灯条检测都只在这块区域里做，缺省为整图；
  //   light_roi 有值时只把中心落在其中的灯条放进 ArmorFrame::lights，无值时
  //             lights 为空；
  //   color     只保留该颜色的灯条，Unknown 表示红蓝都要。
  // 后端或分类器抛出的异常在这里转成一条日志和空结果，不会中断主循环。
  [[nodiscard]] ArmorFrame detectFrame(
    const cv::Mat& image, const std::optional<cv::Rect>& light_roi = std::nullopt,
    const std::optional<cv::Rect>& net_roi = std::nullopt,
    ArmorColor color = ArmorColor::Unknown) const;

  // 网络输入的宽高比（宽 / 高），取自后端的输入形状；后端不可用时返回 1.0。
  // L3 的 netFocusRoi 用它把 ROI 修成同一比例，减少 letterbox 填充。
  [[nodiscard]] double net_aspect_ratio() const noexcept;

  // 最近一帧的调试快照：颜色过滤、合并之后的全部灯条（不受 light_roi 限制），
  // 以及全部配对候选。每次 detectFrame 进来先清空。
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
  LightFinderConfig finder_config_{};
  ImagePreprocessConfig preprocess_config_{};
  // detectFrame 对外是 const，这两个快照只供调试读取，所以用 mutable。
  mutable std::vector<Light> last_lights_;
  mutable std::vector<ArmorCandidate> last_candidates_;
};

}  // namespace L2Perception
