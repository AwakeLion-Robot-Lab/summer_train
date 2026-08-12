#include "l2_perception/armor/armor_detector.hpp"

#include "l6_telemetry/logger.hpp"

#include <exception>
#include <utility>

namespace L2Perception
{

ArmorDetector::ArmorDetector(
  std::unique_ptr<IInferenceBackend> backend,
  ArmorDecoderConfig decoder_config,
  ImagePreprocessConfig preprocess_config,
  ArmorRefinerConfig refiner_config)
  : backend_(std::move(backend))
  , decoder_(std::move(decoder_config))
  , preprocess_config_(std::move(preprocess_config))
  , refiner_(std::move(refiner_config))
{
}

bool ArmorDetector::ready() const noexcept
{
  // 默认构造时 backend_ 为空，表示模型尚未配置，而不是一次推理错误。
  return backend_ != nullptr && backend_->ready();
}

std::vector<Armor> ArmorDetector::detect(const cv::Mat& image) const
{
  last_refine_stats_ = {};
  last_refine_records_.clear();
  if (!ready() || image.empty()) {
    return {};
  }

  try {
    // 这四行就是 L2 神经网络检测完整的职责边界；后续 PnP/Tracker 不应混进这里。
    const PreprocessedImage preprocessed = ImagePreprocessor::run(
      image, backend_->inputSpec(), preprocess_config_);
    const InferenceResult raw_result = backend_->infer(preprocessed.input);
    std::vector<Armor> armors = decoder_.decode(raw_result, preprocessed.transform);
    // 精修必须用原始图像而不是 letterbox 后的模型输入：Decoder 已经把角点还原到
    // 原图坐标，在原图上做 ROI 才不会二次引入缩放误差。批量入口内部逐块调用与
    // SP-Vision 一致的 detect(Armor&, image)，失败项会保留网络角点。
    last_refine_stats_ = refiner_.refine(
      image, armors, collect_refine_records_ ? &last_refine_records_ : nullptr);
    return armors;
  } catch (const std::exception& error) {
    // L2 运行循环不应因一帧坏图或一次推理失败退出；错误留给日志和上层重连策略处理。
    L6Telemetry::logError("armor inference failed", error.what());
    return {};
  }
}

}  // namespace L2Perception
