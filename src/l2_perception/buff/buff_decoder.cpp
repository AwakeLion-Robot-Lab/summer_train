#include "l2_perception/buff/buff_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace L2Perception::Buff {

Decoder::Decoder(DecoderConfig config) : config_(config)
{
  if (!std::isfinite(config_.confidence_threshold) ||
      config_.confidence_threshold <= 0 || config_.confidence_threshold > 1 ||
      !std::isfinite(config_.nms_iou_threshold) ||
      config_.nms_iou_threshold <= 0 || config_.nms_iou_threshold > 1) {
    throw std::invalid_argument("Buff confidence/NMS thresholds must be in (0,1]");
  }
}

std::vector<ModelDetection> Decoder::decode(
  const InferenceResult& result, const ImageTransform& transform,
  L1Sensor::EnemyColor enemy) const
{
  if (result.outputs.size() != 1) {
    throw std::invalid_argument("Buff model requires one [1,16,N] output");
  }
  const auto& tensor = result.outputs.front();
  if (!tensor.isConsistent() || tensor.shape.size() != 3 ||
      tensor.shape[0] != 1 || tensor.shape[1] != 16 || tensor.shape[2] == 0) {
    throw std::invalid_argument("Buff model output must be [1,16,N] (BR/BW/RR/RW, 4x2 keypoints)");
  }
  if (!std::isfinite(transform.source_to_model_scale) ||
      transform.source_to_model_scale <= 0 ||
      transform.source_size.width <= 0 || transform.source_size.height <= 0) {
    throw std::invalid_argument("Invalid Buff image transform");
  }
  if (enemy == L1Sensor::EnemyColor::Unknown) return {};

  const auto values = tensor.values();
  const auto count = tensor.shape[2];
  const auto at = [&](std::size_t channel, std::size_t index) {
    return values[channel * count + index];
  };
  const cv::Rect2f image_rect{0, 0, static_cast<float>(transform.source_size.width),
                                  static_cast<float>(transform.source_size.height)};
  std::vector<ModelDetection> candidates;
  for (std::size_t i = 0; i < count; ++i) {
    bool finite = true;
    for (std::size_t c = 0; c < 16; ++c) finite &= std::isfinite(at(c, i));
    if (!finite) continue;
    int cls = 0;
    for (int c = 1; c < 4; ++c) {
      if (at(4 + c, i) > at(4 + cls, i)) cls = c;
    }
    const auto color = cls < 2 ? L1Sensor::EnemyColor::Blue : L1Sensor::EnemyColor::Red;
    const float confidence = at(4 + cls, i);
    if (color != enemy || confidence < config_.confidence_threshold || confidence > 1 ||
        at(2, i) <= 0 || at(3, i) <= 0) continue;
    const auto tl = transform.modelToSource({at(0, i) - at(2, i) / 2,
                                             at(1, i) - at(3, i) / 2});
    const auto br = transform.modelToSource({at(0, i) + at(2, i) / 2,
                                             at(1, i) + at(3, i) / 2});
    ModelDetection detection;
    detection.box = cv::Rect2f{tl, br} & image_rect;
    if (detection.box.width < 2 || detection.box.height < 2) continue;
    detection.confidence = confidence;
    detection.class_id = cls;
    detection.active = (cls % 2) == 1;
    for (std::size_t k = 0; k < 4; ++k) {
      detection.keypoints[k] = transform.modelToSource({at(8 + k * 2, i), at(9 + k * 2, i)});
      finite &= image_rect.contains(detection.keypoints[k]);
    }
    if (!finite) continue; // 不把图外点夹到边界后拿去算几何。
    // 四点沿周向排列。对角线交点在透视下仍是靶面中心，不能简单平均四点。
    const auto a = detection.keypoints[0];
    const auto u = detection.keypoints[2] - a;
    const auto b = detection.keypoints[1];
    const auto v = detection.keypoints[3] - b;
    const auto cross = [](cv::Point2f x, cv::Point2f y) { return x.x * y.y - x.y * y.x; };
    const float denominator = cross(u, v);
    if (std::abs(denominator) < 1e-3F) continue;
    const float t = cross(b - a, v) / denominator;
    const float s = cross(b - a, u) / denominator;
    if (!(t > 0 && t < 1 && s > 0 && s < 1)) continue;
    detection.center = a + t * u;
    if (!detection.box.contains(detection.center)) continue;
    candidates.push_back(detection);
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.confidence > b.confidence;
  });
  // 每帧最多五叶，限制异常输出的 NMS 代价；同一靶面的 R/W 重复框也要去重。
  if (candidates.size() > 100) candidates.resize(100);
  std::vector<ModelDetection> kept;
  for (const auto& candidate : candidates) {
    const bool duplicate = std::any_of(kept.begin(), kept.end(), [&](const auto& previous) {
      const float intersection = (candidate.box & previous.box).area();
      const float area = candidate.box.area() + previous.box.area() - intersection;
      return area > 0 && intersection / area > config_.nms_iou_threshold;
    });
    if (!duplicate) kept.push_back(candidate);
  }
  return kept;
}

} // namespace L2Perception::Buff
