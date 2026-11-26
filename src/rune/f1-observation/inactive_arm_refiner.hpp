#pragma once

#include "rune/f6_telemetry/types.hpp"

namespace Rune::detail {
void refineInactiveArm(const cv::Mat& bgr, Color color, const Keypoints& current,
                       const Keypoints& raw, const Keypoints* previous, TimePoint now,
                       double max_gap_s, float measurement_weight, DetectedContours& output);
}
