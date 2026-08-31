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
  return detectFrame(image).armors;
}

double ArmorDetector::networkAspectRatio() const noexcept
{
  if (!backend_) {
    return 1.0;
  }
  // 输入契约是 uint8 NHWC：{N, H, W, C}。
  const auto& shape = backend_->inputSpec().shape;
  if (shape.size() < 3 || shape[1] == 0) {
    return 1.0;
  }
  return static_cast<double>(shape[2]) / static_cast<double>(shape[1]);
}

ArmorFrame ArmorDetector::detectFrame(
  const cv::Mat& image, const std::optional<cv::Rect>& light_roi,
  const std::optional<cv::Rect>& net_roi, ArmorColor light_color) const
{
  last_refine_stats_ = {};
  last_refine_records_.clear();
  if (!ready() || image.empty()) {
    return {};
  }

  try {
    // 网络只跑在 ROI 上：远距小目标裁剪后再 resize 到网络输入，相当于局部
    // 放大，保留灯条边缘与数字结构。ROI 缺省或退化为整图时这里是恒等操作。
    const cv::Rect image_rect(0, 0, image.cols, image.rows);
    const cv::Rect focus =
      net_roi ? (*net_roi & image_rect) : image_rect;
    const bool cropped = focus.area() > 0 && focus != image_rect;
    const cv::Mat network_input = cropped ? image(focus) : image;

    // 这四行就是 L2 神经网络检测完整的职责边界；后续 PnP/Tracker 不应混进这里。
    const PreprocessedImage preprocessed = ImagePreprocessor::run(
      network_input, backend_->inputSpec(), preprocess_config_);
    const InferenceResult raw_result = backend_->infer(preprocessed.input);
    ArmorFrame frame;
    frame.armors = decoder_.decode(raw_result, preprocessed.transform);

    // Decoder 把角点还原到了**ROI** 坐标，这里补上偏移回到原图。精修和下游
    // 全部工作在原图坐标系，所以必须在精修之前做。
    if (cropped) {
      const cv::Point2f offset(
        static_cast<float>(focus.x), static_cast<float>(focus.y));
      for (Armor& armor : frame.armors) {
        for (cv::Point2f& corner : armor.corners) {
          corner += offset;
        }
        for (cv::Point2f& corner : armor.network_corners) {
          corner += offset;
        }
        armor.center += offset;
      }
    }
    // 精修必须用原始图像而不是 letterbox 后的模型输入：Decoder 已经把角点还原到
    // 原图坐标，在原图上做 ROI 才不会二次引入缩放误差。批量入口内部逐块调用与
    // SP-Vision 一致的 detect(Armor&, image)，失败项会保留网络角点。
    last_refine_stats_ = refiner_.refine(
      image, frame.armors, collect_refine_records_ ? &last_refine_records_ : nullptr);
    if (light_roi) {
      frame.lights = refiner_.detectLights(
        image, *light_roi, frame.armors, light_color);
    }
    return frame;
  } catch (const std::exception& error) {
    // L2 运行循环不应因一帧坏图或一次推理失败退出；错误留给日志和上层重连策略处理。
    L6Telemetry::logError("armor inference failed", error.what());
    return {};
  }
}

}  // namespace L2Perception
