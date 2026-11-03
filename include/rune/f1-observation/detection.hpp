#pragma once

#include "rune/types.hpp"

#include <array>
#include <cstddef>

namespace Rune {

// Counts describe one successfully decoded frame, before contour/pose processing.
struct DetectorDiagnostics {
  std::size_t anchors{0}, score_pass{0}, keypoint_rejected{0};
  std::size_t nms_rejected{0}, mode_rejected{0}, detections{0};
  std::size_t partial_detections{0}, best_valid_keypoints{0};
  std::size_t fused_candidates{0}; // Suppressed duplicates contributing coordinates.
  std::size_t inference_passes{1}; // Counters otherwise describe the selected pass.
  std::array<float, 3> max_class_scores{};
  int best_class{-1};
  float best_min_keypoint_score{0};
};

// Fill Frame::detections before calling Processor. OpenVinoKeypointDetector
// implements the published five-point model; alternate adapters can implement
// this interface without introducing a model dependency into Processor.
class IKeypointDetector {
public:
  virtual ~IKeypointDetector() = default;
  [[nodiscard]] virtual std::vector<Keypoints> detect(const cv::Mat& bgr) = 0;
  [[nodiscard]] virtual std::vector<Keypoints> detect(const cv::Mat& bgr, Mode mode) {
    (void)mode;
    return detect(bgr);
  }
  [[nodiscard]] virtual std::optional<DetectorDiagnostics> lastDiagnostics() const {
    return std::nullopt;
  }
};

// Network decoding belongs to the caller's model adapter. This stage validates
// its full-image keypoints, chooses an inactive blade and owns a local ROI.
class DetectionStage {
public:
  explicit DetectionStage(double roi_scale = 2.0) : roi_scale_(roi_scale) {}
  [[nodiscard]] std::optional<Region> run(const Frame& frame) const;
  // Includes activated blades as supporting observations, with owned ROIs.
  [[nodiscard]] std::vector<Region> runAll(const Frame& frame) const;

private:
  double roi_scale_;
};

}  // namespace Rune
