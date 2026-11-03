#pragma once

#include "rune/f1-observation/detection.hpp"

#include <string>

namespace Rune {

// Shared by the live debug window and rune_detect output images.
// Draws the four-point target outline, five colored anchors, and class/score.
void drawDetectionOverlay(cv::Mat& image, const std::vector<Keypoints>& detections,
                          Mode mode);

// Current-exposure filtered projections, separate from detector keypoints.
// Missing calibration, stale poses and points behind the camera are skipped.
void drawTrackingOverlay(cv::Mat& image, const Frame& frame, const Pose& pose);

[[nodiscard]] std::string formatDetectionDiagnostics(const DetectorDiagnostics& diagnostics);
void drawDetectionDiagnostics(cv::Mat& image, const DetectorDiagnostics& diagnostics);

}  // namespace Rune
