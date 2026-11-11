#pragma once

#include "rune/f1-observation/detection.hpp"
#include "rune/f4-planning/planning.hpp"

#include <string>

namespace Rune {

// Shared by the live debug window and rune_detect output images.
// Draws inactive circles in blue, activated circles in orange, five colored
// anchors, and class/score.
void drawDetectionOverlay(cv::Mat& image, const std::vector<Keypoints>& detections,
                          Mode mode);

// Current-exposure filtered projections, separate from detector keypoints.
// Missing calibration, stale poses and points behind the camera are skipped.
void drawTrackingOverlay(cv::Mat& image, const Frame& frame, const Pose& pose);

// Draw all five future target circles in green using the fitted motion.
// Uses the current target's ballistic horizon when available; otherwise shows
// an explicitly labelled preview. Unready motion is labelled and not projected.
// Returns true when at least one future target outline was drawn.
bool drawPredictionOverlay(cv::Mat& image, const Frame& frame, const Motion& motion,
                           const Geometry& geometry, const AimPlan* plan = nullptr,
                           double preview_horizon_s = 0.2);

[[nodiscard]] std::string formatDetectionDiagnostics(const DetectorDiagnostics& diagnostics);
void drawDetectionDiagnostics(cv::Mat& image, const DetectorDiagnostics& diagnostics);

}  // namespace Rune
