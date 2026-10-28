#include "l6_telemetry/buff/overlay.hpp"

#include <opencv2/imgproc.hpp>

namespace L6Telemetry::Buff {

void drawOverlay(cv::Mat& image,
                 const L2Perception::Buff::Detections& detections,
                 const L3Estimation::Buff::Addition& addition,
                 const L4Planning::Buff::Plan& plan)
{
  if (image.empty()) return;
  constexpr const char* labels[] = {"BR", "BW", "RR", "RW"};
  for (const auto& detection : detections.model_detections) {
    cv::rectangle(image, detection.box, cv::Scalar{255, 200, 0}, 1);
    cv::putText(image, cv::format("%s %.2f", labels[detection.class_id], detection.confidence),
                detection.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar{255, 200, 0}, 1);
    for (const auto& point : detection.keypoints) {
      cv::circle(image, point, 2, cv::Scalar{255, 200, 0}, cv::FILLED);
    }
  }
  for (const auto& icon : detections.icons) {
    cv::circle(image, icon.center, 4, cv::Scalar{0, 255, 255}, 1);
  }
  for (const auto& bullseye : detections.bullseyes) {
    const cv::Scalar color = bullseye.active
      ? cv::Scalar{0, 255, 0} : cv::Scalar{0, 0, 255};
    cv::circle(image, bullseye.center, 5, color, 1);
    for (const auto& corner : bullseye.corners) {
      cv::circle(image, corner, 2, color, cv::FILLED);
    }
  }
  for (const auto& feature : addition.predicted) {
    cv::drawMarker(image, feature.point, cv::Scalar{255, 128, 0},
                   cv::MARKER_CROSS, 8, 1);
  }
  for (const auto& feature : addition.tracked) {
    cv::drawMarker(image, feature.point, cv::Scalar{255, 255, 255},
                   cv::MARKER_TILTED_CROSS, 8, 1);
  }
  const cv::Scalar plan_color = plan.degraded
    ? cv::Scalar{0, 128, 255} : cv::Scalar{0, 255, 0};
  cv::putText(image,
              plan.valid ? (plan.algorithm_shoot ? "BUFF SHOOT" : "BUFF AIM")
                         : "BUFF LOST",
              cv::Point{20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, plan_color, 2);
  const char* speed_source = plan.bullet_speed.measured
    ? "MCU" : (plan.bullet_speed.trusted ? "FALLBACK TRUSTED" : "FALLBACK NO-SHOOT");
  cv::putText(
    image,
    cv::format("speed %.2f m/s  %s", plan.bullet_speed.value, speed_source),
    cv::Point{20, 58}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
    plan.bullet_speed.trusted ? cv::Scalar{0, 255, 0} : cv::Scalar{0, 165, 255},
    1);
}

}  // namespace L6Telemetry::Buff
