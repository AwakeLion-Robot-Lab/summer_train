#pragma once

#include "rune/types.hpp"

namespace Rune {

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
