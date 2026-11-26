#pragma once

#include "rune/f6_telemetry/types.hpp"

namespace Rune {

// SCUT ellipse-normalized gap measurements and contour strip endpoints.
// Missing features stay absent; no regular polygon is generated as a fallback.
void extractScutFeaturePoints(DetectedContours& measured,
    const cv::RotatedRect& ellipse, cv::Point2f hub, bool inactive,
    const std::vector<std::vector<cv::Point>>& holes,
    const std::vector<std::vector<cv::Point>>& fan_contours);

// radial_reference 给出 R 标到灯臂外侧的方向，允许靶圈不可见。
void extractScutActiveFanPoints(DetectedContours& measured,
    cv::Point2f hub, cv::Point2f radial_reference,
    const std::vector<std::vector<cv::Point>>& fan_contours);

} // namespace Rune
