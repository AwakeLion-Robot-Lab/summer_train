#include "l2_perception/armor/armor_decoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <opencv2/dnn/dnn.hpp>

namespace L2Perception
{
namespace
{

struct DecodedCandidate
{
  Armor detection;
  cv::Rect bounds;
  // SP 默认保存 objectness；显式兼容其他模型时也可保存类别最大分数。
  float nms_score{0.0F};
};

[[nodiscard]] double sigmoid(double value) noexcept
{
  if (value > 0.0) {
    return 1.0 / (1.0 + std::exp(-value));
  }
  const double exp_value = std::exp(value);
  return exp_value / (1.0 + exp_value);
}

[[nodiscard]] cv::Rect boundsOf(const std::array<cv::Point2f, 4>& corners)
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
  // SP-Vision 将浮点关键点直接构造成 cv::Rect，坐标和尺寸在此截断为整数。
  return {static_cast<int>(min_x), static_cast<int>(min_y), static_cast<int>(max_x - min_x),
          static_cast<int>(max_y - min_y)};
}

[[nodiscard]] float iou(const cv::Rect& first, const cv::Rect& second) noexcept
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

ArmorDecoder::ArmorDecoder(ArmorDecoderConfig config) : config_(std::move(config))
{
  std::array<bool, 4> used{};
  for (const std::size_t corner_index : config_.corner_order) {
    if (corner_index >= used.size() || used[corner_index]) {
      throw std::invalid_argument("ArmorDecoder corner_order must be a permutation of 0, 1, 2, 3");
    }
    used[corner_index] = true;
  }
}

std::vector<Armor> ArmorDecoder::decode(const InferenceResult& result,
                                        const ImageTransform& transform) const
{
  const InferenceTensor* output = nullptr;
  if (config_.output_name.empty()) {
    if (result.outputs.size() != 1) {
      throw std::invalid_argument("ArmorDecoder requires output_name when a "
                                  "model has multiple outputs");
    }
    output = &result.outputs.front();
  } else {
    output = result.findOutput(config_.output_name);
    if (output == nullptr) {
      throw std::invalid_argument("ArmorDecoder could not find configured model output: " +
                                  config_.output_name);
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
  const std::size_t required_fields = std::max(
      {config_.corner_offset + 8, config_.confidence_index + 1,
       config_.color_offset + config_.color_count, config_.class_offset + config_.class_count});
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
    double score = valueAt(candidate, config_.confidence_index);
    if (config_.confidence_is_logit) {
      score = sigmoid(score);
    }
    if (!std::isfinite(score) || score < config_.confidence_threshold) {
      continue;
    }
    const float confidence = static_cast<float>(score);

    std::array<cv::Point2f, 4> raw_model_corners{};
    for (std::size_t corner = 0; corner < raw_model_corners.size(); ++corner) {
      float x = valueAt(candidate, config_.corner_offset + corner * 2);
      float y = valueAt(candidate, config_.corner_offset + corner * 2 + 1);
      if (config_.coordinates_are_normalized) {
        x *= static_cast<float>(transform.model_size.width);
        y *= static_cast<float>(transform.model_size.height);
      }
      raw_model_corners[corner] = {x, y};
    }
    if (!finiteCorners(raw_model_corners)) {
      continue;
    }

    Armor detection;
    // SP-Vision 明确按 0、3、2、1 重排模型点。这里不再按几何位置重新排序，
    // 避免强透视或异常点让角点身份发生跳变。
    for (std::size_t corner = 0; corner < detection.corners.size(); ++corner) {
      detection.corners[corner] =
          transform.modelToSource(raw_model_corners[config_.corner_order[corner]]);
      detection.center += detection.corners[corner];
    }
    detection.center = detection.center * 0.25F;
    detection.confidence = confidence;

    // 颜色/类别一般是 logits；比较大小求 argmax 无需先做 softmax。
    if (config_.color_count > 0) {
      std::size_t best_color = 0;
      for (std::size_t color = 1; color < config_.color_count; ++color) {
        if (valueAt(candidate, config_.color_offset + color) >
            valueAt(candidate, config_.color_offset + best_color)) {
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
        if (valueAt(candidate, config_.class_offset + class_index) >
            valueAt(candidate, config_.class_offset + best_class)) {
          best_class = class_index;
        }
      }
      detection.class_id = static_cast<int>(best_class) + config_.class_id_offset;
      best_class_score = valueAt(candidate, config_.class_offset + best_class);
    }

    // SP-Vision 默认以 sigmoid(objectness) 作为 NMS 分数。仍保留 ClassScore
    // 配置项，便于显式兼容其他同形状模型，但它不再是默认行为。
    const float nms_score =
        config_.nms_score_source == ArmorNmsScoreSource::ClassScore ? best_class_score : confidence;
    if (!std::isfinite(nms_score) || nms_score < config_.nms_score_threshold) {
      continue;
    }

    const cv::Rect bounds = boundsOf(detection.corners);
    if (bounds.width <= 0.0F || bounds.height <= 0.0F) {
      continue;
    }
    candidates.push_back({.detection = detection, .bounds = bounds, .nms_score = nms_score});
  }

  std::vector<const DecodedCandidate*> kept_candidates;
  if (!config_.class_aware_nms) {
    // 默认路径直接调用与 SP-Vision 相同的 OpenCV NMSBoxes。
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(candidates.size());
    scores.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      boxes.push_back(candidate.bounds);
      scores.push_back(candidate.nms_score);
    }

    std::vector<int> kept_indices;
    cv::dnn::NMSBoxes(boxes, scores, config_.nms_score_threshold, config_.nms_iou_threshold,
                      kept_indices);
    kept_candidates.reserve(kept_indices.size());
    for (const int index : kept_indices) {
      kept_candidates.push_back(&candidates[static_cast<std::size_t>(index)]);
    }
  } else {
    // 非默认兼容路径：不同颜色/类别分别进行贪心 NMS。
    std::sort(candidates.begin(), candidates.end(),
              [](const DecodedCandidate& left, const DecodedCandidate& right) {
                return left.nms_score > right.nms_score;
              });

    for (const auto& candidate : candidates) {
      bool suppressed = false;
      for (const auto* kept : kept_candidates) {
        if (candidate.detection.class_id != kept->detection.class_id ||
            candidate.detection.color != kept->detection.color) {
          continue;
        }
        if (iou(candidate.bounds, kept->bounds) > config_.nms_iou_threshold) {
          suppressed = true;
          break;
        }
      }
      if (!suppressed) {
        kept_candidates.push_back(&candidate);
      }
    }
  }

  // 先完整执行 NMS，再应用 SP demo 的 min_confidence (> 0.8)。这个顺序与
  // YOLOV5::parse() -> check_name() 一致。
  std::vector<Armor> detections;
  detections.reserve(kept_candidates.size());
  for (const auto* candidate : kept_candidates) {
    if (candidate->detection.confidence > config_.minimum_confidence) {
      detections.push_back(candidate->detection);
    }
  }
  return detections;
}

}  // namespace L2Perception
