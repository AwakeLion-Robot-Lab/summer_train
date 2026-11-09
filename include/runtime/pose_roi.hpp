#pragma once
#include <algorithm>
#include <cmath>
#include <opencv2/core/types.hpp>

namespace runtime {
// Small expansion with approximately preserved aspect ratio (integer rounding);
// clip only at the image boundary.
// It is a heuristic margin, not a guaranteed motion bound. Validate after pose arrival.
inline cv::Rect paddedPoseRoi(cv::Rect roi, cv::Size image) {
  const int dx = std::max(2, static_cast<int>(std::ceil(roi.width * 0.02)));
  const int dy = std::max(2, static_cast<int>(std::ceil(roi.height * 0.02)));
  return cv::Rect(roi.x - dx, roi.y - dy, roi.width + 2 * dx, roi.height + 2 * dy) &
    cv::Rect(0, 0, image.width, image.height);
}
inline bool coversPoseRoi(cv::Rect used, cv::Rect aligned) {
  return aligned.area() > 0 && (used & aligned) == aligned;
}
struct PoseOverlapState {
  int cooldown{0}, consecutive_repairs{0};
  bool beginFrame() {
    if (cooldown > 0) { --cooldown; return false; }
    return true;
  }
  void finishAttempt(bool repaired) {
    consecutive_repairs = repaired ? consecutive_repairs + 1 : 0;
    if (consecutive_repairs >= 2) { cooldown = 30; consecutive_repairs = 0; }
  }
};
} // namespace runtime
