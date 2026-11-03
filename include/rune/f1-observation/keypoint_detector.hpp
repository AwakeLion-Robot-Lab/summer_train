#pragma once

#include "rune/f1-observation/detection.hpp"
#include "armor/l1_perception/inference/inference_backend.hpp"

#include <memory>
#include <string>

namespace YAML { class Node; }

namespace Rune {

struct KeypointDetectorConfig {
  std::filesystem::path model_path{"model/rune_model/model-0624.onnx"};
  std::string device{"CPU"};
  float confidence_threshold{0.8F};
  float keypoint_confidence_threshold{0.8F};
  // SZU demo defaults: confidence=0.8, keypoint confidence=0.8, at least 3 points.
  int min_valid_keypoints{3};
  float nms_distance_px{30.0F};
  // Fraction of the letterboxed input occupied by the fitted image, (0, 1].
  // Smaller values reduce target scale without cropping; points still use source pixels.
  float input_scale{1.0F};
  // Optional second inference only when the primary scale returns no detection.
  // 0 disables retry; coordinates and thresholds keep the same source contract.
  float fallback_input_scale{0.0F};
  // Fuse geometrically consistent duplicates within one frame, after ordinary
  // NMS grouping. Disabled by default to retain the upstream decoding policy.
  bool fuse_keypoints{false};
  float fusion_min_quality_ratio{0.8F};
  // Maximum per-point disagreement relative to the reliable target diameter.
  float fusion_max_point_distance_ratio{0.2F};

  // Load rune.detector from auto_aim.yaml. Explicit relative model paths are
  // resolved from the YAML file's directory.
  [[nodiscard]] static KeypointDetectorConfig load(const std::filesystem::path& yaml);
  [[nodiscard]] static KeypointDetectorConfig fromYaml(
      const YAML::Node& node, const std::filesystem::path& base_dir);
};

// Synchronous, single-thread use. Published head: 3 classes + 5 (x,y,score)
// keypoints, without a bounding box or objectness field.
class OpenVinoKeypointDetector final : public IKeypointDetector {
public:
  explicit OpenVinoKeypointDetector(KeypointDetectorConfig config = {});
  // A loaded backend can be supplied for replay/tests; it must use RGB / 255.
  OpenVinoKeypointDetector(std::unique_ptr<L2Perception::IInferenceBackend> backend,
                          KeypointDetectorConfig config = {});
  [[nodiscard]] std::vector<Keypoints> detect(const cv::Mat& bgr) override;
  [[nodiscard]] std::vector<Keypoints> detect(const cv::Mat& bgr, Mode mode) override;
  [[nodiscard]] std::optional<DetectorDiagnostics> lastDiagnostics() const override {
    return diagnostics_;
  }

private:
  [[nodiscard]] std::vector<Keypoints> detectAtScale(const cv::Mat& bgr, Mode mode,
                                                   float input_scale);
  void validate();
  std::unique_ptr<L2Perception::IInferenceBackend> backend_;
  KeypointDetectorConfig config_;
  std::vector<std::size_t> output_shape_;
  std::string output_name_;
  bool channels_first_{true};
  std::optional<DetectorDiagnostics> diagnostics_;
};

}  // namespace Rune
