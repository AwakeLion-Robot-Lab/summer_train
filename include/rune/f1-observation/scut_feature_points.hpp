#pragma once

#include "rune/types.hpp"

namespace Rune {

// SCUT ellipse-normalized gap measurements and contour strip endpoints.
// Missing features stay absent; no regular polygon is generated as a fallback.
void extractScutFeaturePoints(DetectedContours& measured,
    const cv::RotatedRect& ellipse, cv::Point2f hub, bool inactive,
    const std::vector<std::vector<cv::Point>>& holes,
    const std::vector<std::vector<cv::Point>>& fan_contours);

} // namespace Rune
