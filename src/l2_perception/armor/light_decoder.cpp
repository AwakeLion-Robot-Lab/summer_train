#include "l2_perception/armor/light_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include <opencv2/dnn/dnn.hpp>

namespace L2Perception
{
namespace
{

constexpr const char* kOutputName = "output0";
constexpr int kScoreChannel = 4;
constexpr int kKeypointBase = 5;
constexpr std::size_t kChannels = 11;  // 4 box + 1 score + 2 x (x, y, v)

// 按名字找输出；导出时改过名的单输出模型退回唯一的那个。
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

// 在灯条外扩的一小块 ROI 里比较 R 与 B 的能量，并**跳过饱和像素**。
//
// 颜色不让网络学：过曝时灯条核心是纯白的（R≈G≈B），恰恰不带颜色信息，颜色
// 只留在边缘的光晕里。只沿主轴取核心像素会让 ~70% 的灯条判成未知，实测就是
// 这么掉的。所以取一个比灯条略宽的框，把三通道都接近饱和的像素整个剔掉，
// 剩下的才是有判别力的光晕。
ArmorColor classifyColor(
  const cv::Mat& image, const cv::Point2f& top, const cv::Point2f& bottom,
  double ratio_threshold)
{
  const cv::Point2f center = (top + bottom) * 0.5F;
  const float length = static_cast<float>(cv::norm(top - bottom));
  // 横向按灯条长度的一半外扩：灯条宽约为长的 1/6，这样能吃到两侧光晕。
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
      // 饱和像素没有颜色信息，留着只会把比值往 1 拉。
      if (pixel[0] >= 245 && pixel[1] >= 245 && pixel[2] >= 245) {
        continue;
      }
      // 太暗的背景像素同样无信息，且数量远多于灯条本身。
      if (pixel[0] < 40 && pixel[1] < 40 && pixel[2] < 40) {
        continue;
      }
      blue += pixel[0];
      red += pixel[2];
      ++counted;
    }
  }
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

}  // namespace

LightDecoder::LightDecoder(LightDecoderConfig config) : config_(std::move(config))
{
}

void LightDecoder::validateOutputs(const std::vector<InferenceOutputSpec>& outputs)
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
    throw std::runtime_error(
      "light model: output must be [1, 11, A] (YOLOv8-pose, 1 class, 2 keypoints); got [" +
      shape + "]");
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
    // 端点顺序按图像 y 定，不依赖训练时的关键点编号——云台有 roll，语义定义
    // 会在 roll 大时翻转，图像 y 是确定性的。
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
    light.tilt_angle_deg = static_cast<float>(
      std::atan2(std::abs(top.x - bottom.x), std::max(1e-6F, std::abs(top.y - bottom.y))) *
      180.0 / CV_PI);
    light.score = scores[static_cast<std::size_t>(index)];
    light.color = classifyColor(source, top, bottom, config_.color_ratio_threshold);
    light.id = lights.size();
    lights.push_back(light);
  }
  return lights;
}

}  // namespace L2Perception
