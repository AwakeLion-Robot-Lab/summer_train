#pragma once

#include "rune/f6_telemetry/types.hpp"

namespace Rune::detail {
[[nodiscard]] std::optional<ReobservedTargetRing> reobserveRing(
    const cv::Mat& bgr, Color color, const cv::RotatedRect& search, bool from_shared_motion);
}
