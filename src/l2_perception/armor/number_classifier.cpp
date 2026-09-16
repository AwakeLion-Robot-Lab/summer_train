#include "l2_perception/armor/number_classifier.hpp"

#include <fstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

constexpr const char* kNegativeLabel = "negative";

// 标签字符串 → 车辆类别。label.txt 只认这九行（1 2 3 4 5 outpost guard base
// negative），其余直接报错：映射错一个，整车模型的板数和 PnP 板型就跟着错。
ArmorClass classOfLabel(const std::string& label)
{
  if (label == "1") return ArmorClass::Hero;
  if (label == "2") return ArmorClass::Engineer;
  if (label == "3") return ArmorClass::Infantry3;
  if (label == "4") return ArmorClass::Infantry4;
  if (label == "5") return ArmorClass::Infantry5;
  if (label == "outpost") return ArmorClass::Outpost;
  if (label == "guard") return ArmorClass::Guard;
  // MLP 只有一个 base 类。基地的两类装甲板实物都是小板、都是三板，L3 对两者
  // 的处理完全一样，统一取 BaseSmall。
  if (label == "base") return ArmorClass::BaseSmall;
  if (label == kNegativeLabel) return ArmorClass::Unknown;
  throw std::runtime_error("number classifier: unknown label '" + label + "'");
}

}  // namespace

void NumberClassifier::load(const NumberClassifierConfig& config)
{
  ready_ = false;
  config_ = config;
  labels_.clear();
  classes_.clear();

  if (!std::filesystem::exists(config.model_path)) {
    throw std::runtime_error("number classifier: model not found " + config.model_path.string());
  }
  net_ = cv::dnn::readNetFromONNX(config.model_path.string());
  if (net_.empty()) {
    throw std::runtime_error("number classifier: failed to load " + config.model_path.string());
  }

  std::ifstream file(config.label_path);
  if (!file.is_open()) {
    throw std::runtime_error("number classifier: cannot open " + config.label_path.string());
  }
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      classes_.push_back(classOfLabel(line));
      labels_.push_back(std::move(line));
    }
  }
  if (labels_.empty()) {
    throw std::runtime_error("number classifier: empty label file " + config.label_path.string());
  }

  // 跑一次空输入比对输出维度：标签数与模型对不上时 argmax 的下标会整体错位
  // 成另一辆车，而且不会有任何报错。
  const cv::Mat blank(28, 20, CV_8UC1, cv::Scalar{0});
  net_.setInput(cv::dnn::blobFromImage(blank, 1.0 / 255.0));
  const cv::Mat logits = net_.forward();
  if (logits.total() != labels_.size()) {
    throw std::runtime_error(
      "number classifier: model has " + std::to_string(logits.total()) + " outputs but " +
      config.label_path.string() + " has " + std::to_string(labels_.size()) + " labels");
  }
  ready_ = true;
}

NumberResult NumberClassifier::classify(
  const cv::Mat& bgr, const Light& left, const Light& right, bool large) const
{
  NumberResult result;
  if (!ready_) {
    return result;
  }

  // 透视到固定画布：两灯条的四个端点映到画布两侧，灯条在画布里固定 12 px
  // 长、画布高 28 px，再从中间裁出 20x28。灯条长度归一化之后数字大小也就
  // 归一化了，MLP 才能用这么小的输入。
  constexpr int kLightLength = 12;
  constexpr int kWarpHeight = 28;
  constexpr int kSmallArmorWidth = 32;
  constexpr int kLargeArmorWidth = 54;
  const cv::Size roi_size(20, 28);

  const cv::Point2f lights_vertices[4] = {left.bottom, left.top, right.top, right.bottom};
  const float top_light_y = static_cast<float>((kWarpHeight - kLightLength) / 2 - 1);
  const float bottom_light_y = top_light_y + static_cast<float>(kLightLength);
  const int warp_width = large ? kLargeArmorWidth : kSmallArmorWidth;
  const float right_x = static_cast<float>(warp_width - 1);
  const cv::Point2f target_vertices[4] = {
    {0.0F, bottom_light_y}, {0.0F, top_light_y}, {right_x, top_light_y},
    {right_x, bottom_light_y}};
  const cv::Mat transform = cv::getPerspectiveTransform(lights_vertices, target_vertices);
  cv::Mat number_image;
  cv::warpPerspective(bgr, number_image, transform, cv::Size(warp_width, kWarpHeight));
  number_image =
    number_image(cv::Rect(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size));

  // 入参是 BGR 原图，所以用 BGR2GRAY。写反会把红蓝权重对调，OTSU 之后数字
  // 笔画的粗细跟着变。
  cv::cvtColor(number_image, number_image, cv::COLOR_BGR2GRAY);
  cv::threshold(number_image, number_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

  // 二值图除以 255 得到 0/1 输入，与训练时一致。
  net_.setInput(cv::dnn::blobFromImage(number_image, 1.0 / 255.0));
  const cv::Mat logits = net_.forward().reshape(1, 1);

  double max_logit = 0.0;
  cv::minMaxLoc(logits, nullptr, &max_logit);
  cv::Mat probability;
  cv::exp(logits - max_logit, probability);
  probability /= cv::sum(probability)[0];

  double confidence = 0.0;
  cv::Point class_point;
  cv::minMaxLoc(probability, nullptr, &confidence, nullptr, &class_point);

  result.label_index = class_point.x;
  result.confidence = confidence;
  result.armor_class = classes_[static_cast<std::size_t>(class_point.x)];
  result.number_image = std::move(number_image);

  // 三条拒绝理由按顺序判，分开记以便离线工具统计：分到 negative、置信度不够、
  // 数字对应的板型与配对时按中心距离推出的大小板矛盾。板型由 isLargeArmor
  // 给出，全项目共用同一份映射。
  if (labels_[static_cast<std::size_t>(class_point.x)] == kNegativeLabel) {
    result.verdict = NumberVerdict::Negative;
  } else if (confidence < config_.min_confidence) {
    result.verdict = NumberVerdict::LowConfidence;
  } else if (const auto expected_large = isLargeArmor(result.armor_class);
             expected_large && *expected_large != large) {
    result.verdict = NumberVerdict::TypeMismatch;
  } else {
    result.verdict = NumberVerdict::Accepted;
  }
  return result;
}

const std::string& NumberClassifier::label(int index) const
{
  static const std::string unknown = "?";
  return index >= 0 && static_cast<std::size_t>(index) < labels_.size()
           ? labels_[static_cast<std::size_t>(index)]
           : unknown;
}

}  // namespace L2Perception
