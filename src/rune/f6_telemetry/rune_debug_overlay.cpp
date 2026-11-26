#include "rune/f6_telemetry/debug_overlay.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>

namespace Rune {

std::string formatDetectionDiagnostics(const DetectorDiagnostics& d) {
  std::ostringstream text;
  text << "scut det=" << d.detections << " contours=" << d.contour_count
       << " targets=" << d.target_candidates << " R=" << d.center_candidates
       << " arm_match=" << d.arm_matches << " geometry_drop=" << d.geometry_rejected
       << " gaps=" << d.scut_gap_points << " fan_pts=" << d.scut_fan_points
       << " passes=" << d.contour_passes;
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
const cv::Scalar kPredictionColor{0, 190, 255};
constexpr std::array<int, 4> kTargetOrder{0, 1, 3, 2}; // top, left, bottom, right

bool visiblePoint(const cv::Mat& image, const cv::Point2f& point) {
  return std::isfinite(point.x) && std::isfinite(point.y) &&
      point.x >= 0 && point.y >= 0 && point.x < image.cols && point.y < image.rows;
}

void drawAnchor(cv::Mat& image, const cv::Point2f& point, const std::string& label,
                const cv::Scalar& color = kAnchorColor) {
  cv::circle(image, point, 3, color, 1, cv::LINE_AA);
  int baseline = 0;
  const auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
  const int min_y = std::min(size.height, image.rows - 1);
  const cv::Point origin{
      std::clamp(cvRound(point.x) + 5, 0, std::max(0, image.cols - size.width - 1)),
      std::clamp(cvRound(point.y) - 5, min_y, std::max(min_y, image.rows - baseline - 1))};
  cv::putText(image, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.4,
              {255, 255, 255}, 1, cv::LINE_AA);
}

void drawScutFeatures(cv::Mat& image, const DetectedContours& measured, bool active) {
  if (measured.target_ellipse) {
    cv::ellipse(image, *measured.target_ellipse, kAnchorColor, 1, cv::LINE_AA);
    if (visiblePoint(image, measured.target_ellipse->center))
      drawAnchor(image, measured.target_ellipse->center, "0");
  }
  for (std::size_t i = 0; i < measured.gap_points.size(); ++i) {
    const auto& observed = measured.gap_points[i];
    const auto& p = observed ? observed : measured.gap_display_predictions[i];
    if (p && visiblePoint(image, *p))
      drawAnchor(image, *p, std::to_string(i + 1), observed ? kAnchorColor : kPredictionColor);
  }
  auto points = measured.fan_display_predictions;
  std::array<bool,6> observed{};
  const std::size_t slots = active ? 6 : 4;
  for (std::size_t i=0; i<measured.fan_points.size(); ++i) {
    const int id=measured.fanPointId(i);
    if (id<0 || std::size_t(id)>=slots || !measured.fanPointValid(i)) continue;
    points[id]=measured.fan_points[i];
    observed[id]=true;
  }
  for (std::size_t id=0; id<slots; ++id) {
    const auto& point = points[id];
    if (!point || !visiblePoint(image,*point)) continue;
    const auto next=(id+1)%slots;
    const auto& adjacent=points[next];
    if (active && adjacent && visiblePoint(image,*adjacent))
      cv::line(image,*point,*adjacent,observed[id] && observed[next] ? kAnchorColor : kPredictionColor,
               1,cv::LINE_AA);
    drawAnchor(image,*point,std::to_string(id),observed[id] ? kAnchorColor : kPredictionColor);
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
      drawScutFeatures(image, *detection.image_contours, detection.class_id == 1);
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

void drawDetectionOverlay(cv::Mat& image, const Frame& frame) {
  if (image.empty()) return;
  for (const auto& predicted : frame.display_outline_predictions) {
    cv::ellipse(image,predicted.target,kPredictionColor,1,cv::LINE_AA);
    if (predicted.class_id!=1) continue;
    const auto slots=predicted.arm_points.size();
    for (std::size_t i=0;i<slots;++i) {
      const auto& a=predicted.arm_points[i];
      const auto& b=predicted.arm_points[(i+1)%slots];
      if (a && b && visiblePoint(image,*a) && visiblePoint(image,*b))
        cv::line(image,*a,*b,kPredictionColor,1,cv::LINE_AA);
    }
  }
  const cv::Scalar reobserved_color{255,255,0};
  for (const auto& ring : frame.reobserved_target_rings) {
    cv::ellipse(image,ring.ellipse,reobserved_color,1,cv::LINE_AA);
    if (visiblePoint(image,ring.ellipse.center))
      drawAnchor(image,ring.ellipse.center,"ring",reobserved_color);
  }
  drawDetectionOverlay(image,frame.detections,frame.mode);
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
