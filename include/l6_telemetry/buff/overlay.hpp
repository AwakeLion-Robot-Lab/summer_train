#pragma once

#include "l2_perception/buff/types.hpp"
#include "l3_estimation/buff/types.hpp"
#include "l4_planning/buff/types.hpp"

#include <opencv2/core/mat.hpp>

namespace L6Telemetry::Buff {

// 打符调试叠加层只画在传入图像上，不改变检测、跟踪和控制数据。
void drawOverlay(cv::Mat& image,
                 const L2Perception::Buff::Detections& detections,
                 const L3Estimation::Buff::Addition& addition,
                 const L4Planning::Buff::Plan& plan);

}  // namespace L6Telemetry::Buff
