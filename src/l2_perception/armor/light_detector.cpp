#include "l2_perception/armor/light_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

constexpr const char* kOutputName = "output0";
constexpr int kScoreChannel = 4;
constexpr int kKeypointBase = 5;
constexpr std::size_t kChannels = 11;  // 4 box + 1 score + 2 x (x, y, v)

// 按名字取输出；导出时改过名的单输出模型退回唯一的那个，多输出且没有 output0
// 时返回 nullptr 交给调用方报错。
template <typename Output>
const Output* pickOutput(const std::vector<Output>& outputs)
{
  for (const Output& output : outputs) {
    if (output.name == kOutputName) {
      return &output;
    }
  }
  return outputs.size() == 1 ? &outputs.front() : nullptr;
}

// 端点连线与图像竖直方向的夹角，单位为度。分母取绝对值下限避免灯条水平时除零。
float tiltDegrees(const cv::Point2f& top, const cv::Point2f& bottom)
{
  return static_cast<float>(
    std::atan2(std::abs(top.x - bottom.x), std::max(1e-6F, std::abs(top.y - bottom.y))) *
    180.0 / CV_PI);
}

}  // namespace

std::string_view lightModeName(LightMode mode) noexcept
{
  switch (mode) {
    case LightMode::Model:
      return "model";
    case LightMode::Classic:
      return "classic";
    case LightMode::Hybrid:
      return "hybrid";
  }
  return "unknown";
}

std::optional<LightMode> parseLightMode(std::string_view name) noexcept
{
  for (const LightMode mode : {LightMode::Model, LightMode::Classic, LightMode::Hybrid}) {
    if (name == lightModeName(mode)) {
      return mode;
    }
  }
  return std::nullopt;
}

ArmorColor lightColor(
  const cv::Mat& image, const cv::Point2f& top, const cv::Point2f& bottom,
  double ratio_threshold)
{
  // 取样框：横向按灯条长度的 0.45 倍外扩，灯条宽约为长的 1/6，这样能吃到两侧光晕。
  const cv::Point2f center = (top + bottom) * 0.5F;
  const float length = static_cast<float>(cv::norm(top - bottom));
  const float half_width = std::max(3.0F, length * 0.45F);
  const float half_height = std::max(3.0F, length * 0.60F);
  cv::Rect roi(
    static_cast<int>(std::lround(center.x - half_width)),
    static_cast<int>(std::lround(center.y - half_height)),
    static_cast<int>(std::lround(half_width * 2.0F)),
    static_cast<int>(std::lround(half_height * 2.0F)));
  roi &= cv::Rect(0, 0, image.cols, image.rows);
  if (roi.width < 3 || roi.height < 3) {
    return ArmorColor::Unknown;
  }

  double red = 0.0;
  double blue = 0.0;
  long long counted = 0;
  const cv::Mat patch = image(roi);
  for (int y = 0; y < patch.rows; ++y) {
    const cv::Vec3b* row = patch.ptr<cv::Vec3b>(y);
    for (int x = 0; x < patch.cols; ++x) {
      const cv::Vec3b& pixel = row[x];
      // 三通道都接近饱和的像素没有颜色信息，留着只会把比值往 1 拉。
      if (pixel[0] >= 245 && pixel[1] >= 245 && pixel[2] >= 245) {
        continue;
      }
      // 同理丢掉过暗的背景像素，它们数量远多于灯条本身。
      if (pixel[0] < 40 && pixel[1] < 40 && pixel[2] < 40) {
        continue;
      }
      blue += pixel[0];
      red += pixel[2];
      ++counted;
    }
  }
  // 样本太少时不猜颜色，交给上层当未知灯条丢弃。
  if (counted < 8) {
    return ArmorColor::Unknown;
  }

  const double ratio = red / std::max(1.0, blue);
  if (ratio > ratio_threshold) {
    return ArmorColor::Red;
  }
  if (ratio < 1.0 / ratio_threshold) {
    return ArmorColor::Blue;
  }
  return ArmorColor::Unknown;
}

LightDecoder::LightDecoder(LightDecoderConfig config) : config_(std::move(config))
{
}

void LightDecoder::validate(const std::vector<InferenceOutputSpec>& outputs)
{
  const InferenceOutputSpec* output = pickOutput(outputs);
  if (output == nullptr) {
    throw std::runtime_error(
      "light model: expected an output named output0 or exactly one output");
  }
  if (output->shape.size() != 3 || output->shape[0] != 1 || output->shape[1] != kChannels) {
    std::string shape;
    for (const std::size_t dimension : output->shape) {
      shape += (shape.empty() ? "" : ",") + std::to_string(dimension);
    }
    // 最常见的来源是整板模型和灯条模型的路径填反了，报错里直接点明两者各归哪项。
    throw std::runtime_error(
      "light model: output must be [1, 11, A] (YOLOv8-pose, 1 class, 2 keypoints); got [" +
      shape + "]. light_finder.model_path takes a light keypoint model "
      "(model/light_model/*); whole-armor models (model/armor_model/*) go in "
      "inference.model_path");
  }
}

std::vector<Light> LightDecoder::decode(
  const InferenceResult& result, const ImageTransform& transform,
  const cv::Mat& source) const
{
  const InferenceTensor* tensor = pickOutput(result.outputs);
  if (tensor == nullptr || tensor->shape.size() != 3 || tensor->shape[1] != kChannels ||
      !tensor->isConsistent()) {
    throw std::runtime_error("light model: output does not match the validated contract");
  }

  const std::size_t anchors = tensor->shape[2];
  const std::span<const float> values = tensor->values();
  // channels-first：第 c 通道第 a 个 anchor 在 values[c * anchors + a]。
  const auto at = [&](int channel, std::size_t anchor) {
    return values[static_cast<std::size_t>(channel) * anchors + anchor];
  };

  // 先按分数筛出候选框，picked 记住每个候选对应的 anchor，NMS 之后才好回查关键点。
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<std::size_t> picked;
  for (std::size_t anchor = 0; anchor < anchors; ++anchor) {
    const float score = at(kScoreChannel, anchor);
    if (!(score >= config_.score_threshold)) {
      continue;
    }
    const float cx = at(0, anchor);
    const float cy = at(1, anchor);
    const float width = at(2, anchor);
    const float height = at(3, anchor);
    boxes.emplace_back(
      static_cast<int>(std::lround(cx - width * 0.5F)),
      static_cast<int>(std::lround(cy - height * 0.5F)),
      static_cast<int>(std::lround(width)), static_cast<int>(std::lround(height)));
    scores.push_back(score);
    picked.push_back(anchor);
  }

  std::vector<int> keep;
  if (!boxes.empty()) {
    cv::dnn::NMSBoxes(boxes, scores, config_.score_threshold, config_.nms_iou_threshold, keep);
  }

  std::vector<Light> lights;
  lights.reserve(keep.size());
  for (const int index : keep) {
    const std::size_t anchor = picked[static_cast<std::size_t>(index)];
    // 两个关键点映射回 source 坐标系后，按图像 y 定上下，不按训练时的关键点
    // 编号：云台有 roll 时语义上的「上端点」会翻到下面，图像 y 是确定的。
    cv::Point2f top = transform.modelToSource(
      {at(kKeypointBase + 0, anchor), at(kKeypointBase + 1, anchor)});
    cv::Point2f bottom = transform.modelToSource(
      {at(kKeypointBase + 3, anchor), at(kKeypointBase + 4, anchor)});
    if (bottom.y < top.y) {
      std::swap(top, bottom);
    }

    Light light;
    light.top = top;
    light.bottom = bottom;
    light.center = (top + bottom) * 0.5F;
    light.length = cv::norm(top - bottom);
    light.tilt_angle_deg = tiltDegrees(top, bottom);
    light.score = scores[static_cast<std::size_t>(index)];
    light.source = LightSource::Model;
    light.color = lightColor(source, top, bottom, config_.color_ratio_threshold);
    light.id = lights.size();
    lights.push_back(light);
  }
  return lights;
}

std::vector<Light> findLights(
  const cv::Mat& image, const cv::Rect& roi, const LightFinderConfig& config,
  double color_ratio_threshold, ArmorColor color)
{
  const cv::Rect area = roi & cv::Rect(0, 0, image.cols, image.rows);
  if (image.empty() || image.type() != CV_8UC3 || area.area() <= 0) {
    return {};
  }

  // 底图与阈值必须配套，见 LightFinderConfig 的两个键。
  const bool color_diff = config.color_channel_diff && color != ArmorColor::Unknown;
  cv::Mat base;
  if (color_diff) {
    cv::Mat blue;
    cv::Mat red;
    cv::extractChannel(image(area), blue, 0);
    cv::extractChannel(image(area), red, 2);
    // 饱和减法：白色背景和过曝白核都被压到 0。
    if (color == ArmorColor::Blue) {
      cv::subtract(blue, red, base);
    } else {
      cv::subtract(red, blue, base);
    }
  } else {
    cv::cvtColor(image(area), base, cv::COLOR_BGR2GRAY);
  }
  cv::Mat binary;
  cv::threshold(
    base, binary, color_diff ? config.color_diff_threshold : config.binary_threshold, 255,
    cv::THRESH_BINARY);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  const cv::Point2f offset(static_cast<float>(area.x), static_cast<float>(area.y));
  std::vector<Light> lights;
  for (const std::vector<cv::Point>& contour : contours) {
    // 点数太少的轮廓拟合不出可信的方向。
    if (contour.size() < 6) {
      continue;
    }
    // 角点按 y 排序后，前两个是上边、后两个是下边，取中点作端点并补 roi 偏移。
    const cv::RotatedRect rect = cv::minAreaRect(contour);
    std::array<cv::Point2f, 4> corners;
    rect.points(corners.data());
    std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
      return a.y < b.y;
    });
    const cv::Point2f top = (corners[0] + corners[1]) * 0.5F + offset;
    const cv::Point2f bottom = (corners[2] + corners[3]) * 0.5F + offset;
    const float length = static_cast<float>(cv::norm(top - bottom));
    if (!(length >= config.min_length)) {
      continue;
    }
    // 宽也按排序后的角点取，不用外接矩形的尺寸：横躺的细条按 y 排序后「宽」
    // 是它的长边，比值远大于 1 会被挡掉；用尺寸算则会把它当成竖直的短灯条。
    const float ratio = static_cast<float>(cv::norm(corners[0] - corners[1])) / length;
    if (!(config.min_ratio < ratio && ratio < config.max_ratio)) {
      continue;
    }
    const float tilt_deg = tiltDegrees(top, bottom);
    if (!(tilt_deg < config.max_angle_deg)) {
      continue;
    }

    Light light;
    light.top = top;
    light.bottom = bottom;
    light.center = (top + bottom) * 0.5F;
    light.length = length;
    light.tilt_angle_deg = tilt_deg;
    // 传统检测没有分数，记 1 以免下游按分数排序时排在模型灯条后面。
    light.score = 1.0F;
    light.source = LightSource::Classic;
    light.color = lightColor(image, top, bottom, color_ratio_threshold);
    light.id = lights.size();
    lights.push_back(light);
  }
  return lights;
}

std::vector<Light> mergeLights(
  std::vector<Light> classic, const std::vector<Light>& model, float merge_radius,
  float length_agree)
{
  // 只和传入时就在的传统灯条比，不和后面追加进来的模型灯条比：模型输出已经过 NMS。
  const std::size_t classic_count = classic.size();
  std::vector<bool> replaced(classic_count, false);
  for (const Light& candidate : model) {
    std::size_t nearest = classic_count;
    double nearest_distance = 0.0;
    for (std::size_t index = 0; index < classic_count; ++index) {
      const Light& existing = classic[index];
      const double distance = cv::norm(candidate.center - existing.center);
      const double reach = merge_radius * std::max(existing.length, candidate.length);
      if (distance < reach && (nearest == classic_count || distance < nearest_distance)) {
        nearest = index;
        nearest_distance = distance;
      }
    }
    if (nearest == classic_count) {
      classic.push_back(candidate);
      continue;
    }
    // 判成同一根但长度对不上：二值化把灯条断成了碎块，或者和光晕、背景粘在
    // 一起，这根传统灯条的端点不能用，换成模型的。每根最多被换一次。
    const double shorter = std::min(classic[nearest].length, candidate.length);
    const double longer = std::max(classic[nearest].length, candidate.length);
    if (!replaced[nearest] && longer > 0.0 && shorter / longer < length_agree) {
      classic[nearest] = candidate;
      replaced[nearest] = true;
    }
  }
  return classic;
}

}  // namespace L2Perception
