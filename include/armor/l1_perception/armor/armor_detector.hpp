#pragma once

#include "l1_perception/armor/armor_decoder.hpp"
#include "l1_perception/armor/armor_refiner.hpp"
#include "l1_perception/armor/light_detector.hpp"
#include "l1_perception/inference/inference_backend.hpp"
#include "l1_perception/inference/image_preprocessor.hpp"

#include <memory>
#include <optional>
#include <vector>
#include <opencv2/core.hpp>

namespace L2Perception {

// 整板网络解码后精修角点并保留类别；侧边灯条由 L3 预测位置引导剖面搜索。
struct DetectTiming {
  double preprocess{0.0};
  double infer{0.0};
  double decode{0.0};
  double refine{0.0};
  double side_light{0.0};
};

struct ArmorDetectorConfig {
  ArmorDecoderConfig decoder{};
  LightFinderConfig finder{};
  ImagePreprocessConfig preprocess{};
  ArmorRefinerConfig refiner{};
};

class ArmorDetector {
public:
  ArmorDetector() = default;
  // 启动时校验网络输出形状；模型只加载一次。
  ArmorDetector(std::unique_ptr<IInferenceBackend> armor_backend, ArmorDetectorConfig config);
  bool ready() const noexcept;
  [[nodiscard]] std::vector<Armor> detect(const cv::Mat& image) const;
  // net_roi 缺省为整图。hints 为空时不做侧边灯条搜索；返回原图坐标。
  // color 只筛侧边灯条，整板颜色由调用方筛选。推理失败返回空结果并清空诊断。
  [[nodiscard]] ArmorFrame detectFrame(
      const cv::Mat& image, const std::optional<cv::Rect>& net_roi = std::nullopt,
      ArmorColor color = ArmorColor::Unknown, const std::vector<LightHint>& hints = {}) const;
  [[nodiscard]] double net_aspect_ratio() const noexcept;
  const DetectTiming& lastTiming() const noexcept { return last_timing_; }
  const std::vector<Light>& lastLights() const noexcept { return last_lights_; }
  const ArmorDecoderConfig& decoderConfig() const noexcept { return decoder_.config(); }
  const RefineStats& lastRefineStats() const noexcept { return last_refine_stats_; }

private:
  std::vector<Light> findSideLights(
      const cv::Mat& image, const std::vector<LightHint>& hints, ArmorColor color) const;
  std::unique_ptr<IInferenceBackend> armor_backend_;
  ArmorDecoder decoder_{};
  LightFinderConfig finder_config_{};
  ImagePreprocessConfig preprocess_config_{};
  ArmorRefiner refiner_{};
  mutable RefineStats last_refine_stats_{};
  mutable DetectTiming last_timing_{};
  mutable std::vector<Light> last_lights_;
};

[[nodiscard]] bool insideArmor(
    const Light& light, const Armor& armor, float margin_by_length);

}  // namespace L2Perception
