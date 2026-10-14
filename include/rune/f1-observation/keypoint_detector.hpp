#pragma once

#include "rune/f1-observation/detection.hpp"
#include "armor/l1_perception/inference/inference_backend.hpp"

#include <memory>
#include <string>

namespace Rune {

struct KeypointDetectorConfig {
  std::filesystem::path model_path{"model/rune_model/model-0624.onnx"};
  std::string device{"CPU"};
  float confidence_threshold{0.65F};
  float keypoint_confidence_threshold{0.5F};
  float nms_distance_px{30.0F};

  // Relative model paths are resolved from the YAML file's directory.
  [[nodiscard]] static KeypointDetectorConfig load(const std::filesystem::path& yaml);
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

private:
  void validate();
  std::unique_ptr<L2Perception::IInferenceBackend> backend_;
  KeypointDetectorConfig config_;
  std::vector<std::size_t> output_shape_;
  std::string output_name_;
  bool channels_first_{true};
};

}  // namespace Rune
