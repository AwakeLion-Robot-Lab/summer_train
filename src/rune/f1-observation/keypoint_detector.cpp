// Adapted from SZURPVision/RP-26Rune NNDetector.cpp (MIT).
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
}  // namespace

KeypointDetectorConfig KeypointDetectorConfig::load(const std::filesystem::path& yaml) {
  const auto node = YAML::LoadFile(yaml.string());
  KeypointDetectorConfig config;
  config.model_path = node["model_path"].as<std::string>();
  config.device = node["device"].as<std::string>(config.device);
  config.confidence_threshold = node["confidence_threshold"].as<float>(config.confidence_threshold);
  config.keypoint_confidence_threshold =
    node["keypoint_confidence_threshold"].as<float>(config.keypoint_confidence_threshold);
  config.nms_distance_px = node["nms_distance_px"].as<float>(config.nms_distance_px);
  if (config.model_path.is_relative())
    config.model_path = (yaml.parent_path() / config.model_path).lexically_normal();
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
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("rune detector expects a nonempty CV_8UC3 BGR image");
  if (mode != Mode::Small && mode != Mode::Big)
    throw std::invalid_argument("invalid rune detector mode");

  // Keep upstream's float scale, round-to-nearest resize and centered 114 padding.
  // The armor preprocessor truncates resize dimensions, so it cannot be reused here.
  const auto& spec = backend_->inputSpec();
  const int height = static_cast<int>(spec.shape[1]);
  const int width = static_cast<int>(spec.shape[2]);
  const float scale = std::min(static_cast<float>(width) / bgr.cols,
                               static_cast<float>(height) / bgr.rows);
  const int resized_width = std::clamp(static_cast<int>(std::round(bgr.cols * scale)), 1, width);
  const int resized_height = std::clamp(static_cast<int>(std::round(bgr.rows * scale)), 1, height);
  const int pad_x = (width - resized_width) / 2;
  const int pad_y = (height - resized_height) / 2;
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(height) * width * 3);
  cv::Mat letterbox(height, width, CV_8UC3, pixels.data());
  letterbox.setTo(cv::Scalar(114, 114, 114));
  cv::Mat content = letterbox(cv::Rect(pad_x, pad_y, resized_width, resized_height));
  if (content.size() == bgr.size()) bgr.copyTo(content);
  else cv::resize(bgr, content, content.size(), 0, 0, cv::INTER_LINEAR);
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
  for (std::size_t anchor = 0; anchor < anchors; ++anchor) {
    int best_class = -1;
    float confidence = 0;
    for (int cls = 0; cls < 3; ++cls) {
      const float score = at(cls, anchor);
      if (std::isfinite(score) && score > confidence) {
        confidence = score;
        best_class = cls;
      }
    }
    if (best_class < 0 || confidence < config_.confidence_threshold) continue;
    std::array<cv::Point2f, 5> points;
    cv::Point2f center{};
    float score_sum = 0;
    bool valid = true;
    for (std::size_t point = 0; point < points.size(); ++point) {
      const auto base = 3 + point * 3;
      const float x = (at(base, anchor) - pad_x) / scale;
      const float y = (at(base + 1, anchor) - pad_y) / scale;
      const float score = at(base + 2, anchor);
      // All five points are required by the local PnP/observation contract.
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(score) ||
          x < 0 || y < 0 || score < config_.keypoint_confidence_threshold) {
        valid = false;
        break;
      }
      points[point] = {std::min(x, static_cast<float>(bgr.cols - 1)),
                       std::min(y, static_cast<float>(bgr.rows - 1))};
      score_sum += score;
      center += points[point];
    }
    if (!valid) continue;
    // Model order is top, left, R, right, bottom; local order differs.
    Keypoints keypoints{points[0], points[1], points[3], points[4], points[2],
                        best_class == 0 ? 0 : 1, confidence};
    candidates.push_back({keypoints, center * 0.2F, confidence * score_sum * 0.2F, best_class});
  }
  std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.quality > b.quality;
  });
  std::vector<Candidate> kept;
  const float distance_squared = config_.nms_distance_px * config_.nms_distance_px;
  for (const auto& candidate : candidates) {
    const bool duplicate = std::any_of(kept.begin(), kept.end(), [&](const auto& previous) {
      const auto delta = candidate.center - previous.center;
      return delta.dot(delta) < distance_squared;
    });
    if (!duplicate) kept.push_back(candidate);
  }
  // Match upstream: suppress across all classes first, then filter by mode.
  std::vector<Keypoints> detections;
  for (const auto& candidate : kept) {
    if (candidate.model_class == 0 ||
        candidate.model_class == (mode == Mode::Small ? 1 : 2))
      detections.push_back(candidate.keypoints);
  }
  return detections;
}

}  // namespace Rune
