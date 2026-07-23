#pragma once

#include "l2_perception/armor/armor_decoder.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"

#include <memory>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// L2 装甲检测编排层：图像预处理 → 原始推理 → 装甲模型解码。
// 它不拥有 PnP、跟踪或开火策略；这些工作在 L3/L4/L5。
class ArmorDetector
{
public:
  // 默认构造表示“未配置模型”，detect() 会安全返回空结果，便于 runtime 先启动相机和串口。
  ArmorDetector() = default;
  ArmorDetector(
    std::unique_ptr<IInferenceBackend> backend,
    ArmorDecoderConfig decoder_config = {},
    ImagePreprocessConfig preprocess_config = {});

  [[nodiscard]] bool ready() const noexcept;
  // 一帧同步检测。Backend/Decoder 抛出的异常会被转换为日志和空结果，避免中断主循环。
  [[nodiscard]] std::vector<Armor> detect(const cv::Mat& image) const;

private:
  std::unique_ptr<IInferenceBackend> backend_;
  ArmorDecoder decoder_{};
  ImagePreprocessConfig preprocess_config_{};
};

}  // namespace L2Perception
