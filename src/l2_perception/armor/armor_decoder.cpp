#include "l2_perception/armor/armor_decoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace L2Perception
{
namespace
{

struct DecodedCandidate
{
  Armor detection;
  cv::Rect2f bounds;
  // 使用类别最大分数做 NMS 的排序和二次阈值过滤，保留该值以复现模型原行为。
  float nms_score{0.0F};
};

[[nodiscard]] float sigmoid(float value) noexcept
{
  if (value >= 0.0F) {
    const float exp_value = std::exp(-value);
    return 1.0F / (1.0F + exp_value);
  }

  const float exp_value = std::exp(value);
  return exp_value / (1.0F + exp_value);
}

[[nodiscard]] std::array<cv::Point2f, 4> sortCorners(std::array<cv::Point2f, 4> corners)
{
  // 模型不一定保证角点顺序。先按相对中心的角度环绕排序，再旋转到左上角起点。
  // 对图像 y 向下的坐标系，结果为左上、右上、右下、左下，满足 PnP 约定。
  cv::Point2f center{};
  for (const auto& corner : corners) {
    center += corner;
  }
  center *= 0.25F;

  std::sort(corners.begin(), corners.end(), [&center](const cv::Point2f& left, const cv::Point2f& right) {
    const float left_angle = std::atan2(left.y - center.y, left.x - center.x);
    const float right_angle = std::atan2(right.y - center.y, right.x - center.x);
    return left_angle < right_angle;
  });

  const auto top_left = std::min_element(corners.begin(), corners.end(), [](const cv::Point2f& left, const cv::Point2f& right) {
    return left.x + left.y < right.x + right.y;
  });
  std::rotate(corners.begin(), top_left, corners.end());
  return corners;
}

[[nodiscard]] cv::Rect2f boundsOf(const std::array<cv::Point2f, 4>& corners)
{
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();
  for (const auto& corner : corners) {
    min_x = std::min(min_x, corner.x);
    min_y = std::min(min_y, corner.y);
    max_x = std::max(max_x, corner.x);
    max_y = std::max(max_y, corner.y);
  }
  return {min_x, min_y, max_x - min_x, max_y - min_y};
}

[[nodiscard]] float iou(const cv::Rect2f& first, const cv::Rect2f& second) noexcept
{
  const float intersection = (first & second).area();
  const float union_area = first.area() + second.area() - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

[[nodiscard]] bool finiteCorners(const std::array<cv::Point2f, 4>& corners) noexcept
{
  return std::all_of(corners.begin(), corners.end(), [](const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y);
  });
}

}  // namespace

ArmorDecoder::ArmorDecoder(ArmorDecoderConfig config)
  : config_(std::move(config))
{
}

std::vector<Armor> ArmorDecoder::decode(
  const InferenceResult& result,
  const ImageTransform& transform) const
{
  const InferenceTensor* output = nullptr;
  if (config_.output_name.empty()) {
    if (result.outputs.size() != 1) {
      throw std::invalid_argument("ArmorDecoder requires output_name when a model has multiple outputs");
    }
    output = &result.outputs.front();
  } else {
    output = result.findOutput(config_.output_name);
    if (output == nullptr) {
      throw std::invalid_argument("ArmorDecoder could not find configured model output: " + config_.output_name);
    }
  }

  // Backend 只保证“名字、shape、float 数据”。从这里开始才按装甲模型契约解释它。
  if (!output->isConsistent()) {
    throw std::invalid_argument("ArmorDecoder received an inconsistent output tensor");
  }
  if (output->shape.size() != 2 && output->shape.size() != 3) {
    throw std::invalid_argument("ArmorDecoder only supports [N, F] or [1, N, F] output tensors");
  }
  if (output->shape.size() == 3 && output->shape[0] != 1) {
    throw std::invalid_argument("ArmorDecoder only supports batch size one");
  }

  const std::size_t shape_offset = output->shape.size() == 3 ? 1 : 0;
  const std::size_t candidate_count = config_.tensor_layout == ArmorTensorLayout::CandidatesByFields
                                      ? output->shape[shape_offset]
                                      : output->shape[shape_offset + 1];
  const std::size_t field_count = config_.tensor_layout == ArmorTensorLayout::CandidatesByFields
                                  ? output->shape[shape_offset + 1]
                                  : output->shape[shape_offset];
  const std::size_t required_fields = std::max({
    config_.corner_offset + 8,
    config_.confidence_index + 1,
    config_.color_offset + config_.color_count,
    config_.class_offset + config_.class_count});
  if (field_count < required_fields) {
    throw std::invalid_argument("ArmorDecoder configuration exceeds the model output field count");
  }

  const std::span<const float> output_values = output->values();
  // 将 [candidate, field] 统一映射到连续数据下标，屏蔽两种常见导出布局差异。
  const auto valueAt = [&](std::size_t candidate, std::size_t field) {
    if (config_.tensor_layout == ArmorTensorLayout::CandidatesByFields) {
      return output_values[candidate * field_count + field];
    }
    return output_values[field * candidate_count + candidate];
  };

  std::vector<DecodedCandidate> candidates;
  candidates.reserve(candidate_count);
  for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
    // 先筛低置信度候选，减少后续角点转换和 NMS 的工作量。
    float confidence = valueAt(candidate, config_.confidence_index);
    if (config_.confidence_is_logit) {
      confidence = sigmoid(confidence);
    }
    if (!std::isfinite(confidence) || confidence < config_.confidence_threshold) {
      continue;
    }

    std::array<cv::Point2f, 4> model_corners{};
    for (std::size_t corner = 0; corner < model_corners.size(); ++corner) {
      float x = valueAt(candidate, config_.corner_offset + corner * 2);
      float y = valueAt(candidate, config_.corner_offset + corner * 2 + 1);
      if (config_.coordinates_are_normalized) {
        x *= static_cast<float>(transform.model_size.width);
        y *= static_cast<float>(transform.model_size.height);
      }
      model_corners[corner] = {x, y};
    }
    if (!finiteCorners(model_corners)) {
      continue;
    }

    Armor detection;
    // 先在模型坐标排序，再通过同帧 letterbox 变换还原到原图坐标。
    detection.corners = sortCorners(model_corners);
    for (auto& corner : detection.corners) {
      corner = transform.modelToSource(corner);
    }
    detection.confidence = confidence;

    // 颜色/类别一般是 logits；比较大小求 argmax 无需先做 softmax。
    if (config_.color_count > 0) {
      std::size_t best_color = 0;
      for (std::size_t color = 1; color < config_.color_count; ++color) {
        if (valueAt(candidate, config_.color_offset + color)
            > valueAt(candidate, config_.color_offset + best_color)) {
          best_color = color;
        }
      }
      if (static_cast<int>(best_color) == config_.red_color_index) {
        detection.color = ArmorColor::Red;
      } else if (static_cast<int>(best_color) == config_.blue_color_index) {
        detection.color = ArmorColor::Blue;
      }
    }

    float best_class_score = confidence;
    if (config_.class_count > 0) {
      std::size_t best_class = 0;
      for (std::size_t class_index = 1; class_index < config_.class_count; ++class_index) {
        if (valueAt(candidate, config_.class_offset + class_index)
            > valueAt(candidate, config_.class_offset + best_class)) {
          best_class = class_index;
        }
      }
      detection.class_id = static_cast<int>(best_class) + config_.class_id_offset;
      best_class_score = valueAt(candidate, config_.class_offset + best_class);
    }

    // NMSBoxes 不是按 objectness 排序，而是按最大类别分数排序；
    // 同时类别分数低于 0.65 的候选会被 NMSBoxes 丢弃。这里显式复现该行为。
    const float nms_score = config_.nms_score_source == ArmorNmsScoreSource::ClassScore
                              ? best_class_score
                              : confidence;
    if (!std::isfinite(nms_score) || nms_score < config_.nms_score_threshold) {
      continue;
    }

    const cv::Rect2f bounds = boundsOf(detection.corners);
    if (bounds.width <= 0.0F || bounds.height <= 0.0F) {
      continue;
    }
    candidates.push_back({.detection = detection, .bounds = bounds, .nms_score = nms_score});
  }

  // 贪心 NMS：高置信度框优先保留，后续高度重叠候选被抑制。
  std::sort(candidates.begin(), candidates.end(), [](const DecodedCandidate& left, const DecodedCandidate& right) {
    return left.nms_score > right.nms_score;
  });

  std::vector<Armor> detections;
  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& kept : detections) {
      if (config_.class_aware_nms
          && (candidate.detection.class_id != kept.class_id || candidate.detection.color != kept.color)) {
        continue;
      }
      if (iou(candidate.bounds, boundsOf(kept.corners)) > config_.nms_iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      detections.push_back(candidate.detection);
    }
  }
  return detections;
}

}  // namespace L2Perception
