#include "l2_perception/armor/armor_detector.hpp"

#include "l6_telemetry/logger.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace L2Perception
{

ArmorDetector::ArmorDetector(
  std::unique_ptr<IInferenceBackend> backend, NumberClassifier classifier,
  LightDecoderConfig decoder_config, LightMatcherConfig matcher_config,
  ImagePreprocessConfig preprocess_config)
  : backend_(std::move(backend))
  , classifier_(std::move(classifier))
  , decoder_(std::move(decoder_config))
  , matcher_config_(std::move(matcher_config))
  , preprocess_config_(std::move(preprocess_config))
{
  if (backend_ == nullptr || !backend_->ready()) {
    throw std::invalid_argument("ArmorDetector: light model backend is not loaded");
  }
  if (!classifier_.ready()) {
    throw std::invalid_argument("ArmorDetector: number classifier is not loaded");
  }
  LightDecoder::validateOutputs(probeOutputSpecs(*backend_));
}

bool ArmorDetector::ready() const noexcept
{
  // 默认构造时 backend_ 为空，表示模型尚未配置，而不是一次推理错误。
  return backend_ != nullptr && backend_->ready() && classifier_.ready();
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
  const std::optional<cv::Rect>& net_roi, ArmorColor color) const
{
  last_lights_.clear();
  last_candidates_.clear();
  if (!ready() || image.empty() || image.type() != CV_8UC3) {
    return {};
  }

  try {
    // 网络只跑在 ROI 上：远距小目标裁剪后再 resize 到网络输入，相当于局部
    // 放大，保留灯条端点与数字结构。ROI 缺省或退化为整图时这里是恒等操作。
    const cv::Rect image_rect(0, 0, image.cols, image.rows);
    const cv::Rect focus = net_roi ? (*net_roi & image_rect) : image_rect;
    const bool cropped = focus.area() > 0 && focus != image_rect;
    const cv::Mat network_input = cropped ? image(focus) : image;

    const PreprocessedImage preprocessed = ImagePreprocessor::run(
      network_input, backend_->inputSpec(), preprocess_config_);
    const InferenceResult raw_result = backend_->infer(preprocessed.input);
    std::vector<Light> lights =
      decoder_.decode(raw_result, preprocessed.transform, network_input);

    // 解码结果在 ROI 坐标里，补上偏移回到原图。配对抠数字和下游全部工作在
    // 原图坐标系，所以必须先做。
    if (cropped) {
      const cv::Point2f offset(static_cast<float>(focus.x), static_cast<float>(focus.y));
      for (Light& light : lights) {
        light.top += offset;
        light.bottom += offset;
        light.center += offset;
      }
    }
    std::erase_if(lights, [color](const Light& light) {
      return light.color == ArmorColor::Unknown ||
             (color != ArmorColor::Unknown && light.color != color);
    });
    for (std::size_t index = 0; index < lights.size(); ++index) {
      lights[index].id = index;
    }

    ArmorFrame frame;
    const std::vector<LightPair> pairs = matchLights(lights, color, matcher_config_);
    last_candidates_.reserve(pairs.size());
    for (const LightPair& pair : pairs) {
      const Light& left = lights[pair.left];
      const Light& right = lights[pair.right];
      NumberResult number = classifier_.classify(image, left, right, pair.large);
      if (number.verdict == NumberVerdict::Accepted) {
        Armor armor;
        armor.corners = {left.top, right.top, right.bottom, left.bottom};
        armor.center = (left.top + right.top + right.bottom + left.bottom) * 0.25F;
        armor.class_id = static_cast<int>(number.armor_class);
        armor.color = left.color;
        armor.confidence = static_cast<float>(number.confidence);
        frame.armors.push_back(armor);
      }
      last_candidates_.push_back({pair, std::move(number)});
    }

    if (light_roi) {
      const cv::Rect roi = *light_roi & image_rect;
      for (const Light& light : lights) {
        if (roi.contains(light.center)) {
          frame.lights.push_back(light);
        }
      }
    }
    last_lights_ = std::move(lights);
    return frame;
  } catch (const std::exception& error) {
    // L2 运行循环不应因一帧坏图或一次推理失败退出；错误留给日志和上层重连策略处理。
    L6Telemetry::logError("armor inference failed", error.what());
    last_lights_.clear();
    last_candidates_.clear();
    return {};
  }
}

}  // namespace L2Perception
