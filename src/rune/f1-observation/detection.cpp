#include "rune/f1-observation/detection.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace Rune {

std::optional<Region> DetectionStage::run(const Frame& frame) const {
  if (frame.bgr.empty() || frame.bgr.type() != CV_8UC3 ||
      !std::isfinite(roi_scale_) || roi_scale_ < 1.0) return std::nullopt;

  const Keypoints* best = nullptr;
  for (const auto& candidate : frame.detections) {
    if (candidate.class_id != 0 || !std::isfinite(candidate.confidence)) continue;
    const std::array<cv::Point2f, 5> points{
      candidate.top, candidate.left, candidate.right, candidate.bottom,
      candidate.center_r};
    bool valid = true;
    for (const auto& point : points) {
      valid &= std::isfinite(point.x) && std::isfinite(point.y) &&
               point.x >= 0 && point.y >= 0 &&
               point.x < frame.bgr.cols && point.y < frame.bgr.rows;
    }
    if (valid && (!best || candidate.confidence > best->confidence)) best = &candidate;
  }
  if (!best) return std::nullopt;

  const std::array<cv::Point2f, 5> points{
    best->top, best->left, best->right, best->bottom, best->center_r};
  float min_x = points[0].x, max_x = min_x, min_y = points[0].y, max_y = min_y;
  for (const auto& point : points) {
    min_x = std::min(min_x, point.x); max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y); max_y = std::max(max_y, point.y);
  }
  if (max_x - min_x < 2 || max_y - min_y < 2) return std::nullopt;
  const double cx = (min_x + max_x) / 2.0;
  const double cy = (min_y + max_y) / 2.0;
  const double half_w = (max_x - min_x) * roi_scale_ / 2.0;
  const double half_h = (max_y - min_y) * roi_scale_ / 2.0;
  const cv::Rect image(0, 0, frame.bgr.cols, frame.bgr.rows);
  const cv::Rect wanted(static_cast<int>(std::floor(cx - half_w)),
                        static_cast<int>(std::floor(cy - half_h)),
                        static_cast<int>(std::ceil(2 * half_w)),
                        static_cast<int>(std::ceil(2 * half_h)));
  const cv::Rect bounds = wanted & image;
  if (bounds.empty()) return std::nullopt;
  return Region{*best, bounds, frame.bgr(bounds).clone()};
}

}  // namespace Rune
