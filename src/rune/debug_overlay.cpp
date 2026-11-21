#include "rune/debug_overlay.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

namespace Rune {

std::string formatDetectionDiagnostics(const DetectorDiagnostics& d) {
  std::ostringstream text;
  if (d.contour_backend) {
    text << "scut det=" << d.detections << " contours=" << d.contour_count
         << " targets=" << d.target_candidates << " R=" << d.center_candidates
         << " arm_match=" << d.arm_matches << " geometry_drop=" << d.geometry_rejected
         << " gaps=" << d.scut_gap_points << " fan_pts=" << d.scut_fan_points
         << " passes=" << d.contour_passes;
    return text.str();
  }
  text << std::fixed << std::setprecision(2)
       << "det=" << d.detections << " score_pass=" << d.score_pass
       << " kpt_drop=" << d.keypoint_rejected << " nms_drop=" << d.nms_rejected
       << " mode_drop=" << d.mode_rejected << " max_cls="
       << d.max_class_scores[0] << '/' << d.max_class_scores[1] << '/'
       << d.max_class_scores[2] << " best_class=" << d.best_class
       << " min_kpt=" << d.best_min_keypoint_score
       << " valid_kpt=" << d.best_valid_keypoints << "/5"
       << " partial=" << d.partial_detections
       << " fused=" << d.fused_candidates << " passes=" << d.inference_passes
       << " ring_ok=" << d.target_ring_refined << " ring_reject=" << d.target_ring_rejected
       << " ring_subpixel=" << d.target_ring_subpixel_refined
       << " ring_reconnect=" << d.target_ring_reconnected
       << " ring_no_contour=" << d.target_ring_no_contour;
  return text.str();
}

void drawDetectionDiagnostics(cv::Mat& image, const DetectorDiagnostics& diagnostics,
                              const cv::Scalar& color) {
  if (image.empty()) return;
  const auto text = formatDetectionDiagnostics(diagnostics);
  int baseline = 0;
  const auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
  const double scale = 0.5 * std::min(1.0, std::max(1, image.cols - 40) / double(size.width));
  cv::putText(image, text, {20, 70}, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(image, text, {20, 70}, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}

namespace {
const cv::Scalar kAnchorColor{128, 255, 128};
constexpr std::array<int, 4> kTargetOrder{0, 1, 3, 2}; // top, left, bottom, right

bool visiblePoint(const cv::Mat& image, const cv::Point2f& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
      point.x >= 0 && point.y >= 0 && point.x < image.cols && point.y < image.rows;
}

void drawAnchor(cv::Mat& image, const cv::Point2f& point, const std::string& label) {
  cv::circle(image, point, 3, kAnchorColor, 1, cv::LINE_AA);
  int baseline = 0;
  const auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
  const int min_y = std::min(size.height, image.rows - 1);
  const cv::Point origin{
      std::clamp(cvRound(point.x) + 5, 0, std::max(0, image.cols - size.width - 1)),
      std::clamp(cvRound(point.y) - 5, min_y, std::max(min_y, image.rows - baseline - 1))};
  cv::putText(image, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.4,
              {255, 255, 255}, 1, cv::LINE_AA);
}

void drawScutFeatures(cv::Mat& image, const DetectedContours& measured) {
  if (measured.target_ellipse) {
    cv::ellipse(image, *measured.target_ellipse, kAnchorColor, 1, cv::LINE_AA);
    if (visiblePoint(image, measured.target_ellipse->center))
      drawAnchor(image, measured.target_ellipse->center, "0");
  }
  for (std::size_t i = 0; i < measured.gap_points.size(); ++i)
    if (measured.gap_points[i] && visiblePoint(image, *measured.gap_points[i]))
      drawAnchor(image, *measured.gap_points[i], std::to_string(i + 1));
  const auto& points = measured.fan_points;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto next = (i + 1) % points.size();
    if (measured.fanPointValid(i) && measured.fanPointValid(next) &&
        visiblePoint(image, points[i]) && visiblePoint(image, points[next]))
      cv::line(image, points[i], points[next], kAnchorColor, 1, cv::LINE_AA);
    if (measured.fanPointValid(i) && visiblePoint(image, points[i]))
      drawAnchor(image, points[i], std::to_string(measured.fanPointId(i)));
  }
}
} // namespace

void drawDetectionOverlay(cv::Mat& image, const std::vector<Keypoints>& detections,
                          Mode /*mode*/) {
  if (image.empty()) return;
  for (const auto& detection : detections) {
    if ((detection.class_id != 0 && detection.class_id != 1) ||
        !std::isfinite(detection.confidence)) continue;
    if (detection.image_contours) {
      drawScutFeatures(image, *detection.image_contours);
      if (detection.point_valid[4] && visiblePoint(image, detection.center_r))
        drawAnchor(image, detection.center_r, "R");
      continue;
    }
    // Numbering follows the local anchor order.
    const std::array<cv::Point2f, 5> points{
        detection.top, detection.left, detection.right,
        detection.bottom, detection.center_r};
    std::array<bool, 5> visible{};
    for (std::size_t k = 0; k < points.size(); ++k)
      visible[k] = detection.point_valid[k] && visiblePoint(image, points[k]);
    for (std::size_t k = 0; k < kTargetOrder.size(); ++k) {
      const auto a = kTargetOrder[k];
      const auto b = kTargetOrder[(k + 1) % kTargetOrder.size()];
      // Never close a gap by drawing an edge through an unreliable anchor.
      if (visible[a] && visible[b])
        cv::line(image, points[a], points[b], kAnchorColor, 1, cv::LINE_AA);
    }
    for (std::size_t k = 0; k < points.size(); ++k) {
      if (visible[k]) drawAnchor(image, points[k], k == 4 ? "4 R" : std::to_string(k));
    }
  }
}

void drawTrackingOverlay(cv::Mat& image, const Frame& frame, const Pose& pose) {
  if (image.empty() || image.size() != frame.bgr.size() || !pose.filter_state ||
      pose.captured_at != frame.captured_at || !frame.T_barrel_camera ||
      frame.camera_matrix.rows != 3 || frame.camera_matrix.cols != 3 ||
      !frame.T_barrel_camera->matrix().allFinite()) return;
  try {
    const auto camera_barrel = frame.T_barrel_camera->inverse();
    const std::array<Eigen::Vector3d, 2> points{pose.center_barrel, pose.target_barrel};
    const std::array<const char*, 2> labels{"filtered R", "filtered target"};
    for (std::size_t i = 0; i < points.size(); ++i) {
      const Eigen::Vector3d point = camera_barrel * points[i];
      if (!point.allFinite() || point.z() <= 0.1) continue;
      std::vector<cv::Point2d> pixels;
      cv::projectPoints(std::vector<cv::Point3d>{{point.x(), point.y(), point.z()}},
          cv::Vec3d{}, cv::Vec3d{}, frame.camera_matrix, frame.distortion, pixels);
      const auto& p = pixels.front();
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
          p.x >= image.cols || p.y >= image.rows) continue;
      const cv::Point pixel(cvRound(p.x), cvRound(p.y));
      const cv::Scalar color = i == 0 ? cv::Scalar(255,255,0) : cv::Scalar(255,255,255);
      cv::circle(image, pixel, 8, color, 1, cv::LINE_AA);
      cv::drawMarker(image, pixel, color, cv::MARKER_CROSS, 11, 1, cv::LINE_AA);
      cv::putText(image, labels[i], pixel + cv::Point(11,-9),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);
    }
  } catch (const cv::Exception&) {
    // Debug drawing must not interrupt command processing for bad calibration.
  }
}

}  // namespace Rune
