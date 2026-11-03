// Adapted from SZURPVision/RP-26Rune NNDetector.cpp (MIT).
// Minimum-point filtering and valid-point NMS follow RuneDetectionModel's
// infer/fivepoints_infer.cpp (MIT).
// Licenses and provenance: model/rune_model/README.md.
#include "rune/f1-observation/keypoint_detector.hpp"

#include "armor/l1_perception/inference/backends/openvino_backend.hpp"

#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Rune {
namespace {
void validateConfig(const KeypointDetectorConfig& config) {
  const auto probability = [](float value) {
    return std::isfinite(value) && value >= 0 && value <= 1;
  };
  if (config.model_path.empty() || config.device.empty() ||
      !probability(config.confidence_threshold) ||
      !probability(config.keypoint_confidence_threshold) ||
      config.min_valid_keypoints < 1 || config.min_valid_keypoints > 5 ||
      !std::isfinite(config.input_scale) || config.input_scale <= 0 || config.input_scale > 1 ||
      !probability(config.fallback_input_scale) ||
      !probability(config.fusion_min_quality_ratio) || config.fusion_min_quality_ratio >= 1 ||
      !probability(config.fusion_max_point_distance_ratio) || config.fusion_max_point_distance_ratio <= 0 ||
      !std::isfinite(config.nms_distance_px) || config.nms_distance_px <= 0) {
    throw std::invalid_argument("invalid rune detector model/device/threshold configuration");
  }
}

std::unique_ptr<L2Perception::IInferenceBackend> loadBackend(
    const KeypointDetectorConfig& config) {
  validateConfig(config);
  auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
  backend->load({.model_path = config.model_path, .device = config.device,
                 .model_color_order = L2Perception::ModelColorOrder::Rgb,
                 .normalization_divisor = 255.0F});
  return backend;
}

struct Candidate {
  Keypoints keypoints;
  cv::Point2f center;
  float quality;
  int model_class;
};

std::array<cv::Point2f, 5> pointsOf(const Keypoints& k) {
  return {k.top, k.left, k.right, k.bottom, k.center_r};
}

struct CandidateGroup {
  const Candidate* seed;
  std::vector<const Candidate*> members;
};

Keypoints fuseGroup(const CandidateGroup& group, const KeypointDetectorConfig& config,
                    DetectorDiagnostics& diagnostics) {
  Keypoints result = group.seed->keypoints;
  if (!config.fuse_keypoints || group.members.size() < 2 || group.seed->quality <= 0)
    return result;
  const auto seed_points = pointsOf(result);
  float diameter = 0;
  for (std::size_t i = 0; i < 4; ++i) for (std::size_t j = i + 1; j < 4; ++j) {
    if (result.point_valid[i] && result.point_valid[j])
      diameter = std::max(diameter, static_cast<float>(cv::norm(seed_points[i] - seed_points[j])));
  }
  if (diameter <= 0) return result;
  const double gate = std::min(config.nms_distance_px,
      std::max(2.0F, diameter * config.fusion_max_point_distance_ratio));
  std::array<cv::Point2d, 5> sums{};
  std::array<double, 5> weights{};
  for (const auto* candidate : group.members) {
    if (candidate->model_class != group.seed->model_class) continue;
    const auto& k = candidate->keypoints;
    const auto points = pointsOf(k);
    int common = 0;
    double disagreement = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
      if (!result.point_valid[i] || !k.point_valid[i]) continue;
      ++common;
      disagreement = std::max(disagreement, cv::norm(points[i] - seed_points[i]));
    }
    if (common < 3 || disagreement >= gate) continue;
    const double score_weight = std::max(0.0,
        double(candidate->quality) / group.seed->quality - config.fusion_min_quality_ratio) /
        (1.0 - config.fusion_min_quality_ratio);
    // Taper continuously to zero at either boundary: a candidate entering or
    // leaving the fusion group must not add a full coordinate step.
    const double geometry_weight = 1.0 - disagreement * disagreement / (gate * gate);
    const double weight = score_weight * score_weight * geometry_weight * geometry_weight;
    if (weight <= 0) continue;
    if (candidate != group.seed) ++diagnostics.fused_candidates;
    for (std::size_t i = 0; i < points.size(); ++i) {
      if (!result.point_valid[i] || !k.point_valid[i]) continue;
      const double point_weight = weight * k.point_confidences[i];
      sums[i] += cv::Point2d(points[i]) * point_weight;
      weights[i] += point_weight;
    }
  }
  // Keep the seed's class, scores, validity and weak ROI seeds. Coordinate
  // fusion cannot promote a weak prediction or change the acceptance policy.
  std::array<cv::Point2f*, 5> output{
      &result.top, &result.left, &result.right, &result.bottom, &result.center_r};
  for (std::size_t i = 0; i < output.size(); ++i)
    if (weights[i] > 0) *output[i] = cv::Point2f(sums[i] / weights[i]);
  return result;
}
}  // namespace

KeypointDetectorConfig KeypointDetectorConfig::load(const std::filesystem::path& yaml) {
  const auto root = YAML::LoadFile(yaml.string());
  const auto rune = root["rune"];
  return fromYaml(rune ? rune["detector"] : YAML::Node{}, yaml.parent_path());
}

KeypointDetectorConfig KeypointDetectorConfig::fromYaml(
    const YAML::Node& node, const std::filesystem::path& base_dir) {
  KeypointDetectorConfig config;
  if (!node || node.IsNull()) return config;
  if (!node.IsMap()) throw std::invalid_argument("rune.detector must be a map");
  if (node["model_path"]) {
    config.model_path = node["model_path"].as<std::string>();
    if (!config.model_path.empty() && config.model_path.is_relative())
      config.model_path = (base_dir / config.model_path).lexically_normal();
  }
  config.device = node["device"].as<std::string>(config.device);
  config.confidence_threshold = node["confidence_threshold"].as<float>(config.confidence_threshold);
  config.keypoint_confidence_threshold =
    node["keypoint_confidence_threshold"].as<float>(config.keypoint_confidence_threshold);
  config.nms_distance_px = node["nms_distance_px"].as<float>(config.nms_distance_px);
  config.min_valid_keypoints = node["min_valid_keypoints"].as<int>(config.min_valid_keypoints);
  config.input_scale = node["input_scale"].as<float>(config.input_scale);
  config.fallback_input_scale = node["fallback_input_scale"].as<float>(config.fallback_input_scale);
  config.fuse_keypoints = node["fuse_keypoints"].as<bool>(config.fuse_keypoints);
  config.fusion_min_quality_ratio = node["fusion_min_quality_ratio"].as<float>(config.fusion_min_quality_ratio);
  config.fusion_max_point_distance_ratio = node["fusion_max_point_distance_ratio"].as<float>(config.fusion_max_point_distance_ratio);
  validateConfig(config);
  return config;
}

OpenVinoKeypointDetector::OpenVinoKeypointDetector(KeypointDetectorConfig config)
    : OpenVinoKeypointDetector(loadBackend(config), config) {}

OpenVinoKeypointDetector::OpenVinoKeypointDetector(
    std::unique_ptr<L2Perception::IInferenceBackend> backend, KeypointDetectorConfig config)
    : backend_(std::move(backend)), config_(std::move(config)) {
  validateConfig(config_);
  validate();
}

void OpenVinoKeypointDetector::validate() {
  if (!backend_ || !backend_->ready())
    throw std::invalid_argument("rune detector requires a loaded backend");
  const auto& shape = backend_->inputSpec().shape;
  if (shape.size() != 4 || shape[0] != 1 || shape[3] != 3 ||
      shape[1] == 0 || shape[2] == 0 ||
      shape[1] > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      shape[2] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("rune detector requires static U8 NHWC [1,H,W,3] input");
  const auto outputs = L2Perception::probeOutputSpecs(*backend_);
  if (outputs.size() != 1)
    throw std::runtime_error("rune model must have exactly one output");
  output_shape_ = outputs.front().shape;
  output_name_ = outputs.front().name;
  if (output_shape_.size() != 3 || output_shape_[0] != 1 ||
      !((output_shape_[1] == 18 && output_shape_[2] > 0) ||
        (output_shape_[2] == 18 && output_shape_[1] > 0)))
    throw std::runtime_error("rune head must be [1,18,A] or [1,A,18]: 3 classes + 5x3 keypoints");
  channels_first_ = output_shape_[1] == 18;
}

std::vector<Keypoints> OpenVinoKeypointDetector::detect(const cv::Mat& bgr) {
  return detect(bgr, Mode::Small);
}

std::vector<Keypoints> OpenVinoKeypointDetector::detect(const cv::Mat& bgr, Mode mode) {
  auto detections=detectAtScale(bgr,mode,config_.input_scale);
  if (!detections.empty() || config_.fallback_input_scale<=0 ||
      config_.fallback_input_scale==config_.input_scale) return detections;
  detections=detectAtScale(bgr,mode,config_.fallback_input_scale);
  diagnostics_->inference_passes=2;
  return detections;
}

std::vector<Keypoints> OpenVinoKeypointDetector::detectAtScale(
    const cv::Mat& bgr, Mode mode, float input_scale) {
  diagnostics_.reset();
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("rune detector expects a nonempty CV_8UC3 BGR image");
  if (mode != Mode::Small && mode != Mode::Big)
    throw std::invalid_argument("invalid rune detector mode");

  // Keep upstream's float scale, round-to-nearest resize and centered 114 padding.
  // The armor preprocessor truncates resize dimensions, so it cannot be reused here.
  const auto& spec = backend_->inputSpec();
  const int height = static_cast<int>(spec.shape[1]);
  const int width = static_cast<int>(spec.shape[2]);
  // Near-range recordings can contain much larger targets than this model handles
  // reliably. Reduce their footprint with padding; decode with the same scale.
  const float scale = input_scale *
      std::min(static_cast<float>(width) / bgr.cols,
               static_cast<float>(height) / bgr.rows);
  const int resized_width = std::clamp(static_cast<int>(std::round(bgr.cols * scale)), 1, width);
  const int resized_height = std::clamp(static_cast<int>(std::round(bgr.rows * scale)), 1, height);
  const int pad_x = (width - resized_width) / 2;
  const int pad_y = (height - resized_height) / 2;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(height) * width * 3, 114);
  cv::Mat letterbox(height, width, CV_8UC3, pixels.data());
  cv::Mat content = letterbox(cv::Rect(pad_x, pad_y, resized_width, resized_height));
  if (content.size() == bgr.size()) bgr.copyTo(content);
  else {
    // Strong downsampling of individual LEDs aliases with linear interpolation.
    // Average their source footprint when applying the optional scale reduction;
    // input_scale=1 keeps the upstream preprocessing exactly as before.
    const int interpolation = input_scale < 1.0F ? cv::INTER_AREA : cv::INTER_LINEAR;
    cv::resize(bgr, content, content.size(), 0, 0, interpolation);
  }
  L2Perception::InferenceInput input;
  input.name = spec.name;
  input.shape = spec.shape;
  input.setOwnedData(std::move(pixels));

  const auto result = backend_->infer(input);
  const auto* tensor = result.findOutput(output_name_);
  if (result.outputs.size() != 1 || !tensor || tensor->shape != output_shape_ ||
      !tensor->isConsistent())
    throw std::runtime_error("rune model returned an inconsistent output tensor");
  const auto values = tensor->values();
  const std::size_t anchors = output_shape_[channels_first_ ? 2 : 1];
  const auto at = [&](std::size_t channel, std::size_t anchor) {
    return values[channels_first_ ? channel * anchors + anchor : anchor * 18 + channel];
  };
  std::vector<Candidate> candidates;
  DetectorDiagnostics diagnostics;
  diagnostics.anchors = anchors;
  float best_score = 0;
  for (std::size_t anchor = 0; anchor < anchors; ++anchor) {
    int best_class = -1;
    float confidence = 0;
    for (int cls = 0; cls < 3; ++cls) {
      const float score = at(cls, anchor);
      if (std::isfinite(score))
        diagnostics.max_class_scores[cls] = std::max(diagnostics.max_class_scores[cls], score);
      if (std::isfinite(score) && score > confidence) {
        confidence = score;
        best_class = cls;
      }
    }
    if (confidence > best_score) {
      best_score = confidence;
      diagnostics.best_class = best_class;
      diagnostics.best_valid_keypoints = 0;
      float minimum = std::numeric_limits<float>::infinity();
      for (std::size_t point = 0; point < 5; ++point) {
        const float score = at(5 + point * 3, anchor);
        minimum = std::min(minimum, std::isfinite(score) ? score : 0.0F);
        diagnostics.best_valid_keypoints +=
            std::isfinite(score) && score >= config_.keypoint_confidence_threshold;
      }
      diagnostics.best_min_keypoint_score = minimum;
    }
    if (best_class < 0 || confidence < config_.confidence_threshold) continue;
    ++diagnostics.score_pass;
    std::array<cv::Point2f, 5> points;
    std::array<float, 5> scores;
    std::array<bool, 5> point_valid;
    cv::Point2f center{};
    float score_sum = 0;
    int valid_count = 0;
    bool valid = true;
    for (std::size_t point = 0; point < points.size(); ++point) {
      const auto base = 3 + point * 3;
      const float x = (at(base, anchor) - pad_x) / scale;
      const float y = (at(base + 1, anchor) - pad_y) / scale;
      const float score = at(base + 2, anchor);
      // Match SZU's minimum-valid-points policy; retain weak finite predictions
      // for ROI extraction, and keep their confidence for downstream consumers.
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(score) ||
          x < 0 || y < 0) {
        valid = false;
        break;
      }
      points[point] = {std::min(x, static_cast<float>(bgr.cols - 1)),
                       std::min(y, static_cast<float>(bgr.rows - 1))};
      scores[point] = score;
      point_valid[point] = score >= config_.keypoint_confidence_threshold;
      if (point_valid[point]) {
        ++valid_count;
        score_sum += score;
        center += points[point];
      }
    }
    if (!valid || valid_count < config_.min_valid_keypoints) {
      ++diagnostics.keypoint_rejected;
      continue;
    }
    // Model order is top, left, R, right, bottom; local order differs.
    Keypoints keypoints{points[0], points[1], points[3], points[4], points[2],
                        best_class == 0 ? 0 : 1, confidence};
    keypoints.point_confidences = {scores[0], scores[1], scores[3], scores[4], scores[2]};
    keypoints.point_valid = {point_valid[0], point_valid[1], point_valid[3], point_valid[4], point_valid[2]};
    candidates.push_back({keypoints, center / valid_count, confidence * score_sum / valid_count, best_class});
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.quality > b.quality;
  });
  std::vector<CandidateGroup> kept;
  const float distance_squared = config_.nms_distance_px * config_.nms_distance_px;
  for (const auto& candidate : candidates) {
    const auto duplicate = std::find_if(kept.begin(), kept.end(), [&](const auto& previous) {
      const auto delta = candidate.center - previous.seed->center;
      return delta.dot(delta) < distance_squared;
    });
    if (duplicate == kept.end()) kept.push_back({&candidate, {&candidate}});
    else {
      if (config_.fuse_keypoints) duplicate->members.push_back(&candidate);
      ++diagnostics.nms_rejected;
    }
  }
  // Match upstream: suppress across all classes first, then filter by mode.
  std::vector<Keypoints> detections;
  for (const auto& group : kept) {
    const auto& candidate = *group.seed;
    if (candidate.model_class == 0 ||
        candidate.model_class == (mode == Mode::Small ? 1 : 2)) {
      detections.push_back(fuseGroup(group, config_, diagnostics));
      diagnostics.partial_detections += std::count(candidate.keypoints.point_valid.begin(),
                                                   candidate.keypoints.point_valid.end(), true) < 5;
    }
    else ++diagnostics.mode_rejected;
  }
  diagnostics.detections = detections.size();
  diagnostics_ = diagnostics;
  return detections;
}

}  // namespace Rune
