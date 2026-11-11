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
#include <numbers>
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

    // Size the circle from reliable target anchors; R is not part of the rim.
    const cv::Scalar outline_color = detection.class_id == 0
        ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 165, 255);
    const std::array<int, 5> local_order{0, 1, 4, 2, 3};
    if (const auto center = detection.armorCenter()) {
      double radius = 0;
      for (const auto k : {0, 1, 3, 4}) {
        if (detection.point_valid[local_order[k]])
          radius = std::max(radius, cv::norm(points[k] - *center));
      }
      if (radius >= 1)
        cv::circle(image, *center, cvRound(radius), outline_color, 2, cv::LINE_AA);
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

namespace {
bool canProject(const cv::Mat& image, const Frame& frame, const Pose& pose) {
  return !image.empty() && image.size() == frame.bgr.size() && pose.filter_state &&
      pose.captured_at == frame.captured_at && frame.T_barrel_camera &&
      frame.camera_matrix.rows == 3 && frame.camera_matrix.cols == 3 &&
      frame.T_barrel_camera->matrix().allFinite();
}
void predictionLabel(cv::Mat& image, const std::string& text, bool ready) {
  cv::putText(image, text, {20, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              {0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(image, text, {20, 130}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              ready ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), 1, cv::LINE_AA);
}
} // namespace

bool drawPredictionOverlay(cv::Mat& image, const Frame& frame, const Motion& motion,
                           const Geometry& geometry, const AimPlan* plan,
                           double preview_horizon_s) {
  const auto& pose = motion.pose;
  if (!canProject(image, frame, pose)) return false;
  if (!motion.ready) {
    predictionLabel(image, "Prediction: waiting for motion fit", false);
    return false;
  }
  const bool intercept = plan && plan->valid && !plan->center_hold &&
      plan->captured_at == frame.captured_at;
  const double horizon = intercept ? plan->prediction_horizon_s : preview_horizon_s;
  if (!std::isfinite(horizon) || horizon <= 0) return false;
  const auto phase = motion.phase_delta_rad ? motion.phase_delta_rad(horizon)
      : (frame.mode == Mode::Small && std::isfinite(motion.angular_speed_rad_s)
          ? std::optional<double>(motion.angular_speed_rad_s * horizon) : std::nullopt);
  if (!phase || !std::isfinite(*phase)) {
    predictionLabel(image, "Prediction: motion model unavailable", false);
    return false;
  }
  bool drawn = false;
  try {
    const auto camera_barrel = frame.T_barrel_camera->inverse();
    if (std::isfinite(geometry.radius_m) && geometry.radius_m > 0 &&
        std::isfinite(geometry.armor_half_width_m) && geometry.armor_half_width_m > 0 &&
        std::isfinite(geometry.armor_half_height_m) && geometry.armor_half_height_m > 0 &&
        pose.T_barrel_model.matrix().allFinite()) {
      // Advance every blade by the fitted phase change at the SAME future
      // instant. The model's +y axis points at the current reference target.
      const double r = geometry.radius_m;
      const double w = geometry.armor_half_width_m;
      const double h = geometry.armor_half_height_m;
      const std::array<Eigen::Vector3d, 4> outline{
          Eigen::Vector3d(0, r + h, 0), Eigen::Vector3d(-w, r, 0),
          Eigen::Vector3d(0, r - h, 0), Eigen::Vector3d(w, r, 0)};
      const Eigen::Isometry3d camera_model = camera_barrel * pose.T_barrel_model;
      for (int sector = 0; sector < 5; ++sector) {
        const Eigen::AngleAxisd rotation(*phase + sector * 2 * std::numbers::pi / 5,
                                        Eigen::Vector3d::UnitZ());
        std::vector<cv::Point3d> corners;
        for (const auto& corner : outline) {
          const Eigen::Vector3d point = camera_model * (rotation * corner);
          if (!point.allFinite() || point.z() <= 0.1) break;
          corners.emplace_back(point.x(), point.y(), point.z());
        }
        if (corners.size() != outline.size()) continue;
        const Eigen::Vector3d center = camera_model * (rotation * Eigen::Vector3d(0, r, 0));
        if (!center.allFinite() || center.z() <= 0.1) continue;
        corners.emplace_back(center.x(), center.y(), center.z());
        std::vector<cv::Point2d> pixels;
        cv::projectPoints(corners, cv::Vec3d{}, cv::Vec3d{},
                          frame.camera_matrix, frame.distortion, pixels);
        if (!std::all_of(pixels.begin(), pixels.end(), [](const auto& pixel) {
              return std::isfinite(pixel.x) && std::isfinite(pixel.y) &&
                     std::abs(pixel.x) <= 1e6 && std::abs(pixel.y) <= 1e6;
            })) continue;
        const auto& pixel_center = pixels.back();
        double radius = 0;
        for (std::size_t k = 0; k < outline.size(); ++k)
          radius = std::max(radius, cv::norm(pixels[k] - pixel_center));
        if (radius >= 1 && radius <= 1e6) {
          cv::circle(image, {cvRound(pixel_center.x), cvRound(pixel_center.y)},
                     cvRound(radius), {0, 255, 0}, 2, cv::LINE_AA);
          drawn = true;
        }
      }
    }
  } catch (const cv::Exception&) {
    // Invalid debug calibration must not interrupt command processing.
    return false;
  }
  if (drawn) {
    std::ostringstream label;
    label << "Prediction +" << std::fixed << std::setprecision(0) << horizon * 1000
          << " ms (" << (intercept ? "intercept" : "preview") << ")"
          << " speed=" << std::setprecision(2) << motion.angular_speed_rad_s << " rad/s";
    predictionLabel(image, label.str(), true);
  }
  return drawn;
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
