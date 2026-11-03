// Keypoint palette and label style reference:
// SZURPVision/RuneDetectionModel, infer/fivepoints_infer.cpp (MIT).
// License and provenance: model/rune_model/{LICENSE,UPSTREAM_NOTICE.md}.
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
  text << std::fixed << std::setprecision(2)
       << "det=" << d.detections << " score_pass=" << d.score_pass
       << " kpt_drop=" << d.keypoint_rejected << " nms_drop=" << d.nms_rejected
       << " mode_drop=" << d.mode_rejected << " max_cls="
       << d.max_class_scores[0] << '/' << d.max_class_scores[1] << '/'
       << d.max_class_scores[2] << " best_class=" << d.best_class
       << " min_kpt=" << d.best_min_keypoint_score
       << " valid_kpt=" << d.best_valid_keypoints << "/5"
       << " partial=" << d.partial_detections
       << " fused=" << d.fused_candidates << " passes=" << d.inference_passes;
  return text.str();
}

void drawDetectionDiagnostics(cv::Mat& image, const DetectorDiagnostics& diagnostics) {
  if (image.empty()) return;
  const auto text = formatDetectionDiagnostics(diagnostics);
  int baseline = 0;
  const auto size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
  const double scale = 0.5 * std::min(1.0, std::max(1, image.cols - 40) / double(size.width));
  cv::putText(image, text, {20, 70}, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(image, text, {20, 70}, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 255, 255}, 1, cv::LINE_AA);
}

void drawDetectionOverlay(cv::Mat& image, const std::vector<Keypoints>& detections,
                          Mode mode) {
  if (image.empty()) return;
  // Upstream order is top, left, R, right, bottom; our stored order differs.
  const std::array<cv::Scalar, 5> colors{
      cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0),
      cv::Scalar(255, 0, 255), cv::Scalar(0, 0, 255)};
  for (const auto& detection : detections) {
    if ((detection.class_id != 0 && detection.class_id != 1) ||
        !std::isfinite(detection.confidence)) continue;
    const std::array<cv::Point2f, 5> points{
        detection.top, detection.left, detection.center_r,
        detection.right, detection.bottom};
    if (!std::all_of(points.begin(), points.end(), [&](const auto& p) {
          return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 &&
                 p.x < image.cols && p.y < image.rows;
        })) continue;

    // Walk around the target, without crossing its diagonals or including R.
    const cv::Scalar outline_color = detection.class_id == 0
        ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
    const std::array<int, 5> local_order{0, 1, 4, 2, 3};
    const std::array<int, 4> perimeter{0, 1, 4, 3};
    for (std::size_t j=0;j<perimeter.size();++j) {
      const auto a=perimeter[j], b=perimeter[(j+1)%perimeter.size()];
      if (detection.point_valid[local_order[a]] && detection.point_valid[local_order[b]])
        cv::line(image, points[a], points[b], outline_color, 2, cv::LINE_AA);
    }
    for (std::size_t k = 0; k < points.size(); ++k) {
      if (!detection.point_valid[local_order[k]]) continue;
      cv::circle(image, points[k], 3, colors[k], cv::FILLED);
      cv::circle(image, points[k], 3, {255, 255, 255}, 1);
    }

    const std::string class_name = detection.class_id == 0 ? "inactive"
        : mode == Mode::Small ? "small_activated" : "big_activated";
    const std::string label = class_name + " " + std::to_string(
        static_cast<int>(std::clamp(detection.confidence, 0.0F, 1.0F) * 100)) + "%";
    cv::Point2f center{};
    int valid_count=0;
    for (std::size_t k=0;k<points.size();++k) if (detection.point_valid[local_order[k]]) {
      center+=points[k]; ++valid_count;
    }
    if (!valid_count) continue;
    center/=valid_count;
    int baseline = 0;
    const auto size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    // Keep labels visible when detections are close to an image edge.
    const int x = std::clamp(cvRound(center.x), 2, std::max(2, image.cols - size.width - 2));
    const int y = std::clamp(cvRound(center.y) - 2, size.height + 2,
                             std::max(size.height + 2, image.rows - baseline - 2));
    cv::rectangle(image, {x - 2, y - size.height - 2},
                  {x + size.width + 2, y + baseline + 2}, {0, 0, 0}, cv::FILLED);
    cv::putText(image, label, {x, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                {255, 255, 255}, 1, cv::LINE_AA);
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
