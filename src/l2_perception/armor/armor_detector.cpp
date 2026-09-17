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
  LightFinderConfig finder_config, ImagePreprocessConfig preprocess_config)
  : backend_(std::move(backend))
  , classifier_(std::move(classifier))
  , decoder_(std::move(decoder_config))
  , matcher_config_(std::move(matcher_config))
  , finder_config_(std::move(finder_config))
  , preprocess_config_(std::move(preprocess_config))
{
  if (backend_ == nullptr || !backend_->ready()) {
    throw std::invalid_argument("ArmorDetector: light model backend is not loaded");
  }
  if (!classifier_.ready()) {
    throw std::invalid_argument("ArmorDetector: number classifier is not loaded");
  }
  LightDecoder::validate(probeOutputSpecs(*backend_));
}

bool ArmorDetector::ready() const noexcept
{
  // backend_ 为空是“默认构造、尚未配置模型”，不是推理出错。
  return backend_ != nullptr && backend_->ready() && classifier_.ready();
}

std::vector<Armor> ArmorDetector::detect(const cv::Mat& image) const
{
  return detectFrame(image).armors;
}

double ArmorDetector::net_aspect_ratio() const noexcept
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
    // area 是这一帧两路检测共同的搜索范围：给了 net_roi 就裁剪，缺省或与整图
    // 相同时是恒等操作。裁剪后再 resize 到网络输入相当于局部放大，远距小目标
    // 的灯条端点和数字结构能保住。
    const cv::Rect image_rect(0, 0, image.cols, image.rows);
    const cv::Rect focus = net_roi ? (*net_roi & image_rect) : image_rect;
    const bool cropped = focus.area() > 0 && focus != image_rect;
    const cv::Rect area = cropped ? focus : image_rect;
    // 丢掉判不出颜色的灯条；指定了颜色就只留该颜色。
    const auto keepColor = [color](std::vector<Light>& lights) {
      std::erase_if(lights, [color](const Light& light) {
        return light.color == ArmorColor::Unknown ||
               (color != ArmorColor::Unknown && light.color != color);
      });
    };

    std::vector<Light> lights;
    if (finder_config_.mode != LightMode::Classic) {
      const cv::Mat network_input = image(area);
      const PreprocessedImage preprocessed = ImagePreprocessor::run(
        network_input, backend_->inputSpec(), preprocess_config_);
      const InferenceResult raw_result = backend_->infer(preprocessed.input);
      lights = decoder_.decode(raw_result, preprocessed.transform, network_input);

      // 模型解出的坐标在裁剪图里，补偏移回原图：抠数字和下游都按原图坐标算。
      if (cropped) {
        const cv::Point2f offset(static_cast<float>(focus.x), static_cast<float>(focus.y));
        for (Light& light : lights) {
          light.top += offset;
          light.bottom += offset;
          light.center += offset;
        }
      }
      keepColor(lights);
    }
    if (finder_config_.mode != LightMode::Model) {
      std::vector<Light> classic = findLights(
        image, area, finder_config_, decoder_.config().color_ratio_threshold);
      // 先过颜色再合并：颜色判不出的传统斑点（数字笔画、光晕）先被丢掉，
      // 不会在 mergeLights 里把旁边真正的模型灯条挤掉。
      keepColor(classic);
      lights = mergeLights(
        std::move(classic), lights, finder_config_.merge_radius, finder_config_.length_agree);
    }
    // 合并会改变顺序，重新编号，让 LightPair 里的下标和 id 对得上。
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

    // ArmorFrame::lights 给 L3 做端点观测，按 light_roi 再筛一次。
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
    // 一帧坏图或一次推理失败不该中断主循环，记日志后当这帧没检出。
    L6Telemetry::logError("armor inference failed", error.what());
    last_lights_.clear();
    last_candidates_.clear();
    return {};
  }
}

}  // namespace L2Perception
