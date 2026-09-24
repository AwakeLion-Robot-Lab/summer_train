#include "l2_perception/armor/armor_detector.hpp"

#include "l6_telemetry/logger.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace L2Perception
{

ArmorDetector::ArmorDetector(
  std::unique_ptr<IInferenceBackend> armor_backend, ArmorDetectorConfig config)
  : armor_backend_(std::move(armor_backend))
  , decoder_(std::move(config.decoder))
  , refiner_(std::move(config.refiner))
  , finder_config_(std::move(config.finder))
  , preprocess_config_(std::move(config.preprocess))
{
  if (armor_backend_ == nullptr || !armor_backend_->ready()) {
    throw std::invalid_argument("ArmorDetector: armor model backend is not loaded");
  }
  decoder_.validate(probeOutputSpecs(*armor_backend_));

  // 模型或标签有问题就在启动阶段抛，别等到每帧把类别判成另一辆车。
  if (config.number.enable) {
    classifier_.load(config.number);
  }
}

NumberStats ArmorDetector::classifyNumbers(
  const cv::Mat& image, std::vector<Armor>& armors) const
{
  NumberStats stats;
  if (!classifier_.ready()) {
    return stats;
  }
  const bool drop = classifier_.config().on_reject == RejectPolicy::Drop;
  std::erase_if(armors, [&](Armor & armor) {
    const NumberResult result = classifier_.classify(image, armor.corners);
    switch (result.verdict) {
      case NumberVerdict::Accepted:
        ++stats.accepted;
        break;
      case NumberVerdict::Negative:
        ++stats.negative;
        return drop;
      case NumberVerdict::LowConfidence:
        ++stats.low_confidence;
        return drop;
      case NumberVerdict::TypeMismatch:
        ++stats.type_mismatch;
        return drop;
    }
    // 只有采信时才改写类别；没采信的板类别保持网络的 argmax，class_source
    // 也仍是 Network，离线统计才分得清哪一路给的编号。
    armor.class_id = static_cast<int>(result.armor_class);
    armor.class_source = ClassSource::Number;
    armor.number_confidence = static_cast<float>(result.confidence);
    return false;
  });
  return stats;
}

bool ArmorDetector::ready() const noexcept
{
  // armor_backend_ 为空是“默认构造、尚未配置模型”，不是推理出错。
  return armor_backend_ != nullptr && armor_backend_->ready();
}

std::vector<Armor> ArmorDetector::detect(const cv::Mat& image) const
{
  return detectFrame(image).armors;
}

double ArmorDetector::net_aspect_ratio() const noexcept
{
  if (!armor_backend_) {
    return 1.0;
  }
  // 输入契约是 uint8 NHWC：{N, H, W, C}。
  const auto& shape = armor_backend_->inputSpec().shape;
  if (shape.size() < 3 || shape[1] == 0) {
    return 1.0;
  }
  return static_cast<double>(shape[2]) / static_cast<double>(shape[1]);
}

std::vector<Light> ArmorDetector::findSideLights(
  const cv::Mat& image, const cv::Rect& roi, const std::vector<LightHint>& hints,
  ArmorColor color) const
{
  std::vector<Light> lights = finder_config_.search == LightSearch::Profile
                                ? searchLights(image, hints, finder_config_, color)
                                : findLights(image, roi, finder_config_, color);

  // 丢掉判不出颜色的灯条；指定了颜色就只留该颜色。判不出颜色的多是数字笔画
  // 和光晕，留着会被 L3 当成灯条去关联。
  std::erase_if(lights, [color](const Light& light) {
    return light.color == ArmorColor::Unknown ||
           (color != ArmorColor::Unknown && light.color != color);
  });

  for (std::size_t index = 0; index < lights.size(); ++index) {
    lights[index].id = index;
  }
  return lights;
}

ArmorFrame ArmorDetector::detectFrame(
  const cv::Mat& image, const std::optional<cv::Rect>& light_roi,
  const std::optional<cv::Rect>& net_roi, ArmorColor color,
  const std::vector<LightHint>& hints) const
{
  last_refine_ = {};
  last_numbers_ = {};
  last_records_.clear();
  last_lights_.clear();
  last_timing_ = {};
  if (!ready() || image.empty() || image.type() != CV_8UC3) {
    return {};
  }

  // 逐段计时。用 steady_clock 与全工程一致；开销是每帧十来次 now()，可忽略。
  auto mark = std::chrono::steady_clock::now();
  const auto lap = [&mark]() {
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - mark).count();
    mark = now;
    return ms;
  };

  try {
    // 网络只跑在 net_roi 上：远距小目标裁剪后再 resize 到网络输入，相当于
    // 局部放大，保留灯条边缘与数字结构。缺省或与整图相同时是恒等操作。
    const cv::Rect image_rect(0, 0, image.cols, image.rows);
    const cv::Rect focus = net_roi ? (*net_roi & image_rect) : image_rect;
    const bool cropped = focus.area() > 0 && focus != image_rect;
    const cv::Mat network_input = cropped ? image(focus) : image;

    const PreprocessedImage preprocessed =
      ImagePreprocessor::run(network_input, armor_backend_->inputSpec(), preprocess_config_);
    last_timing_.preprocess = lap();
    const InferenceResult armor_result = armor_backend_->infer(preprocessed.input);
    last_timing_.infer = lap();
    ArmorFrame frame;
    frame.armors = decoder_.decode(armor_result, preprocessed.transform);
    last_timing_.decode = lap();

    // 解码出的角点在裁剪图里，补偏移回原图。精修和下游都在原图坐标系，所以
    // 必须在精修之前做。
    if (cropped) {
      const cv::Point2f offset(static_cast<float>(focus.x), static_cast<float>(focus.y));
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

    // 精修用原图而不是 letterbox 后的网络输入，免得二次引入缩放误差。失败的
    // 板保留网络角点。
    last_refine_ =
      refiner_.refine(image, frame.armors, collect_records_ ? &last_records_ : nullptr);
    last_timing_.refine = lap();

    // 数字二次分类排在精修之后：抠图用的是精修过的角点，数字区域对得更准；
    // 判不出的板在这里就被丢掉，不再进 L3 的关联。
    last_numbers_ = classifyNumbers(image, frame.armors);
    last_timing_.number = lap();

    if (light_roi) {
      const cv::Rect roi = *light_roi & image_rect;
      if (roi.area() > 0) {
        last_lights_ = findSideLights(image, roi, hints, color);
        // 已检出装甲板自己的灯条已经作为板的角点进了观测，不能再以侧边灯条的
        // 身份进一次；不管是不是正在跟踪的那辆车，都不是“侧边”灯条。
        for (const Light& light : last_lights_) {
          const bool owned = std::any_of(
            frame.armors.begin(), frame.armors.end(), [&](const Armor& armor) {
              return insideArmor(light, armor, finder_config_.armor_margin);
            });
          if (!owned) {
            frame.lights.push_back(light);
          }
        }
      }
    }
    last_timing_.side_light = lap();
    return frame;
  } catch (const std::exception& error) {
    // 一帧坏图或一次推理失败不该中断主循环，记日志后当这帧没检出。
    L6Telemetry::logError("armor inference failed", error.what());
    last_refine_ = {};
    last_numbers_ = {};
    last_records_.clear();
    last_lights_.clear();
    last_timing_ = {};
    return {};
  }
}

bool insideArmor(const Light& light, const Armor& armor, float margin_by_length)
{
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  for (const cv::Point2f& corner : armor.corners) {
    min_x = std::min(min_x, corner.x);
    min_y = std::min(min_y, corner.y);
    max_x = std::max(max_x, corner.x);
    max_y = std::max(max_y, corner.y);
  }
  const float margin = margin_by_length * static_cast<float>(light.length);
  return light.center.x >= min_x - margin && light.center.x <= max_x + margin &&
         light.center.y >= min_y - margin && light.center.y <= max_y + margin;
}

}  // namespace L2Perception
