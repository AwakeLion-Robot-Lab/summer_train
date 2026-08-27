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

std::size_t ArmorTensorContract::requiredFieldCount() const noexcept
{
  // objectness 缺失时它不占字段，也不参与这个下界。
  const std::size_t objectness_fields = objectness_index ? *objectness_index + 1 : 0;
  return std::max({corner_offset + 8, objectness_fields, color_offset + color_count,
                   class_offset + class_count});
}

ArmorDecoderConfig yolov5_22DecoderConfig() noexcept
{
  // 默认构造就是这套契约，这里显式写出来是为了和 yolov8_21 并排可读。
  return ArmorDecoderConfig{};
}

ArmorDecoderConfig yolov8_21DecoderConfig() noexcept
{
  ArmorDecoderConfig config;
  auto& contract = config.contract;
  contract.output_name = "output0";
  // [1, 21, 6300]：每一行是一个字段，每一列是一个候选。
  contract.tensor_layout = ArmorTensorLayout::FieldsByCandidates;
  // 行 0~3 颜色、4~12 九类、13~20 四角点；4 + 9 + 8 = 21，没有 objectness。
  contract.color_offset = 0;
  contract.color_count = 4;
  contract.class_offset = 4;
  contract.class_count = 9;
  contract.corner_offset = 13;
  contract.objectness_index.reset();
  // YOLOV8 的分类头已经过 sigmoid，输出就在 0~1，不能再套一次。
  contract.confidence_is_logit = false;

  // 阈值取上游部署库的同名默认值（score_threshold 0.5、NMS IoU 0.2）。
  // v5 的 objectness 和 v8 的类别分数分布不同，不能沿用 0.7/0.8。
  config.confidence_threshold = 0.5F;
  config.nms_score_threshold = 0.5F;
  // 上游没有 NMS 后的第二道门限，取和初筛相同的值使其成为空操作。
  config.minimum_confidence = 0.5F;
  config.nms_iou_threshold = 0.2F;
  return config;
}

std::optional<ArmorDecoderConfig> armorDecoderPreset(std::string_view name)
{
  if (name == "yolov5_22") {
    return yolov5_22DecoderConfig();
  }
  if (name == "yolov8_21") {
    return yolov8_21DecoderConfig();
  }
  return std::nullopt;
}

ArmorDecoderConfig armorDecoderConfigFor(const std::vector<InferenceOutputSpec>& outputs)
{
  // 按输出名认契约。名字对上就保证 decode() 能找到这个张量，字段数不符会在
  // 那里报错，所以这里不再验一遍形状。
  if (outputs.size() == 1) {
    for (const auto& preset : {yolov5_22DecoderConfig(), yolov8_21DecoderConfig()}) {
      if (outputs.front().name == preset.contract.output_name) {
        return preset;
      }
    }
  }
  throw std::runtime_error(
    "unknown armor model output; expected a single 'output' (yolov5_22) or "
    "'output0' (yolov8_21)");
}

ArmorDecoder::ArmorDecoder(ArmorDecoderConfig config) : config_(std::move(config))
{
  std::array<bool, 4> used{};
  for (const std::size_t index : config_.contract.corner_order) {
    if (index >= used.size() || used[index]) {
      throw std::invalid_argument("ArmorDecoder corner_order must be a permutation of 0, 1, 2, 3");
    }
    used[index] = true;
  }
}

std::vector<Armor> ArmorDecoder::decode(const InferenceResult& result,
                                        const ImageTransform& transform) const
{
  const ArmorTensorContract& contract = config_.contract;

  const InferenceTensor* output = nullptr;
  if (contract.output_name.empty()) {
    if (result.outputs.size() != 1) {
      throw std::invalid_argument("ArmorDecoder requires output_name when a "
                                  "model has multiple outputs");
    }
    output = &result.outputs.front();
  } else {
    output = result.findOutput(contract.output_name);
    if (output == nullptr) {
      throw std::invalid_argument("ArmorDecoder could not find configured model output: " +
                                  contract.output_name);
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

  const bool candidates_first = contract.tensor_layout == ArmorTensorLayout::CandidatesByFields;
  const std::size_t shape_offset = output->shape.size() == 3 ? 1 : 0;
  const std::size_t candidate_count =
    candidates_first ? output->shape[shape_offset] : output->shape[shape_offset + 1];
  const std::size_t field_count =
    candidates_first ? output->shape[shape_offset + 1] : output->shape[shape_offset];
  if (field_count < contract.requiredFieldCount()) {
    throw std::invalid_argument("ArmorDecoder contract exceeds the model output field count");
  }

  const std::span<const float> output_values = output->values();
  // 将 [candidate, field] 统一映射到连续数据下标，屏蔽两种常见导出布局差异。
  const auto valueAt = [&](std::size_t candidate, std::size_t field) {
    return candidates_first ? output_values[candidate * field_count + field]
                            : output_values[field * candidate_count + candidate];
  };

  const auto argmaxClass = [&](std::size_t candidate) {
    std::size_t best = 0;
    for (std::size_t index = 1; index < contract.class_count; ++index) {
      if (valueAt(candidate, contract.class_offset + index) >
          valueAt(candidate, contract.class_offset + best)) {
        best = index;
      }
    }
    return best;
  };

  std::vector<DecodedCandidate> candidates;
  candidates.reserve(candidate_count);
  for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
    // 先筛低置信度候选，减少后续角点转换和 NMS 的工作量。没有 objectness 的
    // 模型必须先求类别 argmax 才拿得到置信度；有 objectness 的则把 argmax 推迟
    // 到初筛之后——2.5 万个候选各多做 9 次比较不是可以忽略的开销。
    std::size_t best_class = 0;
    bool class_resolved = false;
    double score = 0.0;
    if (contract.objectness_index) {
      score = valueAt(candidate, *contract.objectness_index);
    } else {
      best_class = argmaxClass(candidate);
      class_resolved = true;
      score = valueAt(candidate, contract.class_offset + best_class);
    }
    if (contract.confidence_is_logit) {
      score = sigmoid(score);
    }
    if (!std::isfinite(score) || score < config_.confidence_threshold) {
      continue;
    }
    const float confidence = static_cast<float>(score);

    std::array<cv::Point2f, 4> raw_model_corners{};
    for (std::size_t corner = 0; corner < raw_model_corners.size(); ++corner) {
      float x = valueAt(candidate, contract.corner_offset + corner * 2);
      float y = valueAt(candidate, contract.corner_offset + corner * 2 + 1);
      if (contract.coordinates_are_normalized) {
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
          transform.modelToSource(raw_model_corners[contract.corner_order[corner]]);
      detection.center += detection.corners[corner];
    }
    detection.center = detection.center * 0.25F;
    detection.confidence = confidence;

    // 颜色/类别一般是 logits；比较大小求 argmax 无需先做 softmax。
    if (contract.color_count > 0) {
      std::size_t best_color = 0;
      for (std::size_t color = 1; color < contract.color_count; ++color) {
        if (valueAt(candidate, contract.color_offset + color) >
            valueAt(candidate, contract.color_offset + best_color)) {
          best_color = color;
        }
      }
      if (static_cast<int>(best_color) == contract.red_color_index) {
        detection.color = ArmorColor::Red;
      } else if (static_cast<int>(best_color) == contract.blue_color_index) {
        detection.color = ArmorColor::Blue;
      }
    }

    // 无 objectness 的模型这里不会重复求 argmax，初筛时已经算过。
    if (!class_resolved) {
      best_class = argmaxClass(candidate);
    }
    detection.class_id = static_cast<int>(best_class) + contract.class_id_offset;

    // NMS 分数就是候选置信度本身。SP 用 sigmoid(objectness)，YOLOV8 用类别
    // 最大分，两者都已经由上面的 confidence 表示，不需要再开一个来源开关。
    if (confidence < config_.nms_score_threshold) {
      continue;
    }
    const float nms_score = confidence;

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
