#pragma once

#include "rune/types.hpp"

namespace Rune {

enum class TargetRingResult { Refined, Rejected, Skipped };
enum class TargetRingRejectReason { None, InvalidInput, NoSupportedContour, InvalidDirections, OutOfImage };
struct TargetRingDiagnostics {
  bool reconnected{false};
  TargetRingRejectReason rejection{TargetRingRejectReason::None};
};

// Fits the outer illuminated target ring in source pixels. radius_ratio is
// measured target radius / R-to-target distance. Does not promote weak points;
// an unverified ring invalidates target anchors while retaining NN ROI seeds.
[[nodiscard]] TargetRingResult refineTargetRing(
    const cv::Mat& bgr, Keypoints& keypoints, Mode mode, double radius_ratio,
    bool subpixel = true, TargetRingDiagnostics* diagnostics = nullptr);

} // namespace Rune
