// Algorithm reference: scutrobotlab/rm_vision_core at 5c8085ae4b5f1f8f75e7c7d157ea60395f118429.
// Independently implemented adapter; see docs/rune_scut_detection.md for mapping.
#include "rune/f1-observation/scut_contour_detector.hpp"
#include "rune/f1-observation/scut_feature_points.hpp"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace Rune {
namespace {
void validate(const ScutContourConfig& c) {
  if (c.close_kernel < 1 || c.close_kernel > 19 || c.close_kernel % 2 != 1)
    throw std::invalid_argument("rune.detector.scut.close_kernel must be an odd integer in [1, 19], got " +
                                std::to_string(c.close_kernel));
  if (c.red_threshold < 1 || c.red_threshold > 254 || c.blue_threshold < 1 ||
      c.blue_threshold > 254 ||
      c.fallback_threshold_offset < 0 || c.fallback_threshold_offset > 100 ||
      c.max_features < 5 || c.max_features > 512 ||
      !std::isfinite(c.min_target_area) || c.min_target_area < 5 ||
      !std::isfinite(c.max_target_area) || c.max_target_area <= c.min_target_area ||
      !std::isfinite(c.min_center_area) || c.min_center_area < 5 ||
      !std::isfinite(c.max_center_area) || c.max_center_area <= c.min_center_area ||
      !std::isfinite(c.min_center_rect_fill) || c.min_center_rect_fill < 0.3 || c.min_center_rect_fill > 1 ||
      !std::isfinite(c.min_center_target_area_ratio) || c.min_center_target_area_ratio < 0 ||
      c.min_center_target_area_ratio >= 0.5 ||
      !std::isfinite(c.max_target_axis_ratio) || c.max_target_axis_ratio < 1 ||
      c.max_target_axis_ratio > 5 || !std::isfinite(c.min_ellipse_fill) ||
      c.min_ellipse_fill < 0.5 || c.min_ellipse_fill > 1 ||
      !std::isfinite(c.max_ellipse_residual) || c.max_ellipse_residual <= 0 || c.max_ellipse_residual > 0.3 ||
      !std::isfinite(c.radius_ratio) || c.radius_ratio <= 0 || c.radius_ratio >= 0.5 ||
      !std::isfinite(c.radius_ratio_tolerance) || c.radius_ratio_tolerance <= 0 ||
      c.radius_ratio_tolerance >= 0.8 || !std::isfinite(c.max_arm_offset_ratio) ||
      c.max_arm_offset_ratio <= 0 || c.max_arm_offset_ratio >= 0.5 ||
      !std::isfinite(c.min_arm_color_ratio) || c.min_arm_color_ratio < 0 || c.min_arm_color_ratio >= 1)
    throw std::invalid_argument("invalid rune.detector.scut contour configuration");
}

struct Feature {
  int index;
  double area;
  cv::Point2f center;
  cv::RotatedRect ellipse;
};

// Ellipse axes give an affine circle normalization, as in SCUT's gap extraction.
cv::Point2f normalized(const cv::RotatedRect& e, cv::Point2f delta) {
  const double a = e.angle * CV_PI / 180;
  return {float(2 * (delta.x * std::cos(a) + delta.y * std::sin(a)) / e.size.width),
          float(2 * (-delta.x * std::sin(a) + delta.y * std::cos(a)) / e.size.height)};
}
cv::Point2f denormalized(const cv::RotatedRect& e, cv::Point2f p) {
  const double a = e.angle * CV_PI / 180;
  const double x = p.x * e.size.width / 2, y = p.y * e.size.height / 2;
  return {float(x * std::cos(a) - y * std::sin(a)),
          float(x * std::sin(a) + y * std::cos(a))};
}
bool inImage(cv::Point2f p, cv::Size size) {
  return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 &&
         p.x < size.width && p.y < size.height;
}

struct Target {
  Feature feature;
  bool inactive;
  bool concentric;
  std::vector<int> holes;
};
struct Match {
  std::size_t target;
  Keypoints keypoints;
  double quality;
  std::vector<int> fan_indices;
};

std::optional<Match> matchTarget(std::size_t target_index, const Target& target,
    const Feature& hub, const std::vector<Feature>& features,
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<std::vector<cv::Point>>& arm_contours,
    const ScutContourConfig& config, cv::Size image_size, Color color) {
  const auto& t = target.feature;
  const auto delta = t.center - hub.center;
  const double distance = cv::norm(delta);
  if (hub.area > t.area * 0.5 || hub.area < t.area * config.min_center_target_area_ratio || distance < 5)
    return {};
  auto radial = normalized(t.ellipse, delta);
  const double normalized_distance = cv::norm(radial);
  if (!std::isfinite(normalized_distance) || normalized_distance < 1) return {};
  const double radius_ratio = 1 / normalized_distance;
  const double error = std::abs(radius_ratio / config.radius_ratio - 1);
  if (error > config.radius_ratio_tolerance) return {};
  radial *= float(1 / normalized_distance);
  // Conjugate ellipse diameters preserve camera foreshortening; an image-space
  // perpendicular would give incorrect tangential anchors in an oblique view.
  const cv::Point2f tangent{-radial.y, radial.x};
  const auto vertical = denormalized(t.ellipse, radial);
  const auto horizontal = denormalized(t.ellipse, tangent);
  const cv::Point2f axis = delta * float(1 / distance);
  int arm_index = -1;
  double arm_area = 0;
  double low = 1, high = 0, lateral_sum = 0;
  int pieces = 0;
  bool elongated = false, active_fan = false;
  std::vector<int> fan_indices;
  for (const auto& f : features) {
    if (f.area < 15 || f.area > 3 * t.area ||
        cv::norm(normalized(t.ellipse,f.center-t.center)) < 1.1)
      continue;
    const auto v = f.center - hub.center;
    const double along = v.dot(axis) / distance;
    const double lateral = std::abs(v.cross(axis)) / distance;
    const double lane = target.concentric ? config.max_arm_offset_ratio
                                         : std::min(config.max_arm_offset_ratio,0.10);
    if (along < 0.14 || along > 0.84 || lateral > lane) continue;
    const auto rect = cv::minAreaRect(arm_contours[f.index]);
    const double width = std::max(rect.size.width, rect.size.height);
    const double height = std::min(rect.size.width, rect.size.height);
    if (height < 1 || width > distance * 0.85 || height > distance * 0.65) continue;
    const double fill = f.area / std::max(1.0F, rect.size.area());
    // SCUT's inactive arrow/bar and active sparse fan shape filters.
    elongated |= width / height >= 2 && width / height <= 7 && fill >= 0.35;
    active_fan |= width / height < 2 && fill >= 0.05 && fill <= 0.65;
    ++pieces;
    fan_indices.push_back(f.index);
    if (f.area > arm_area) { arm_area = f.area; arm_index = f.index; }
    lateral_sum += lateral;
    for (const auto& p : arm_contours[f.index]) {
      const double projection = (cv::Point2f(p) - hub.center).dot(axis) / distance;
      low = std::min(low, projection); high = std::max(high, projection);
    }
  }
  if (pieces == 0 || high - low < 0.16) return {};
  const bool arrow_chain = pieces >= 3;
  const bool inactive_arm = arrow_chain || elongated;
  const bool inactive = target.inactive || (!target.concentric && inactive_arm);
  if (!inactive && !active_fan) {
    // A clipped active fan may consist only of separate narrow strips. Require
    // their measured semantic endpoints instead of rejecting them by the
    // complete-fan aspect ratio or accepting three arbitrary components.
    std::vector<std::vector<cv::Point>> visible_strips;
    for (const auto i : fan_indices) visible_strips.push_back(arm_contours[i]);
    DetectedContours measured;
    extractScutFeaturePoints(measured,t.ellipse,hub.center,false,{},visible_strips);
    active_fan=measured.fan_points.size()>=3;
  }
  if (inactive ? !inactive_arm : !active_fan) return {};
  Keypoints k;
  k.top = t.center + vertical; k.bottom = t.center - vertical;
  k.left = t.center - horizontal; k.right = t.center + horizontal;
  k.center_r = hub.center;
  for (const auto p : {k.top, k.bottom, k.left, k.right, k.center_r})
    if (!inImage(p, image_size)) return {};
  k.class_id = inactive ? 0 : 1;
  const double quality = 1 - 0.5 * error / config.radius_ratio_tolerance -
      0.2 * lateral_sum / (pieces * config.max_arm_offset_ratio);
  k.confidence = float(std::clamp(quality, 0.1, 1.0));
  k.point_confidences.fill(k.confidence);
  k.target_ring_status = TargetRingStatus::Verified;
  auto measured = std::make_shared<DetectedContours>();
  measured->color = color;
  measured->armor = contours[t.index];
  // Keep an actual closed contour for downstream moments/distance fields.
  // Other disconnected arrows contribute to matching, not invented polygon edges.
  measured->arm = arm_contours[arm_index];
  measured->center_r = contours[hub.index];
  k.image_contours = std::move(measured);
  return Match{target_index, std::move(k), quality, std::move(fan_indices)};
}
} // namespace

ScutContourConfig ScutContourConfig::fromYaml(const YAML::Node& node, const YAML::Node& geometry) {
  ScutContourConfig c;
  if (node && !node.IsNull()) {
    if (!node.IsMap()) throw std::invalid_argument("rune.detector.scut must be a map");
#define RUNE_SCUT_READ(name) c.name = node[#name].as<decltype(c.name)>(c.name)
    RUNE_SCUT_READ(red_threshold); RUNE_SCUT_READ(blue_threshold);
    RUNE_SCUT_READ(fallback_threshold_offset);
    RUNE_SCUT_READ(close_kernel); RUNE_SCUT_READ(min_target_area);
    RUNE_SCUT_READ(max_target_area); RUNE_SCUT_READ(min_center_area);
    RUNE_SCUT_READ(max_center_area); RUNE_SCUT_READ(min_center_rect_fill);
    RUNE_SCUT_READ(min_center_target_area_ratio);
    RUNE_SCUT_READ(max_target_axis_ratio);
    RUNE_SCUT_READ(min_ellipse_fill); RUNE_SCUT_READ(max_ellipse_residual);
    RUNE_SCUT_READ(radius_ratio);
    RUNE_SCUT_READ(radius_ratio_tolerance); RUNE_SCUT_READ(max_arm_offset_ratio);
    RUNE_SCUT_READ(min_arm_color_ratio);
    RUNE_SCUT_READ(max_features);
#undef RUNE_SCUT_READ
  }
  if (geometry && geometry.IsMap()) {
    const auto radius = geometry["radius_m"].as<double>(0);
    const auto height = geometry["armor_half_height_m"].as<double>(0);
    if (radius > 0 && height > 0) c.radius_ratio = height / radius;
  }
  validate(c);
  return c;
}

ScutContourDetector::ScutContourDetector(ScutContourConfig config) : config_(config) {
  validate(config_);
}
std::vector<Keypoints> ScutContourDetector::detect(const cv::Mat& bgr) {
  return detect(bgr, Mode::Small, Color::Blue);
}
std::vector<Keypoints> ScutContourDetector::detect(const cv::Mat& bgr, Mode mode) {
  return detect(bgr, mode, Color::Blue);
}
std::vector<Keypoints> ScutContourDetector::detect(const cv::Mat& bgr, Mode mode, Color color) {
  diagnostics_.reset();
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("SCUT rune detector requires a nonempty CV_8UC3 BGR image");
  const int threshold = color == Color::Red ? config_.red_threshold : config_.blue_threshold;
  auto result = detectAtThreshold(bgr, mode, color, threshold);
  if (result.empty() && config_.fallback_threshold_offset > 0 && threshold < 254) {
    result = detectAtThreshold(bgr, mode, color, std::min(254, threshold + config_.fallback_threshold_offset));
    diagnostics_->contour_passes = 2;
  }
  return result;
}
std::vector<Keypoints> ScutContourDetector::detectAtThreshold(
    const cv::Mat& bgr, Mode mode, Color color, int threshold) {
  DetectorDiagnostics d;
  d.contour_backend = true; d.inference_passes = 0; d.contour_passes = 1;
  cv::Mat red, blue, difference, mask;
  cv::extractChannel(bgr, red, 2); cv::extractChannel(bgr, blue, 0);
  if (color == Color::Red) cv::subtract(red, blue, difference);
  else cv::subtract(blue, red, difference);
  cv::GaussianBlur(difference, difference, {3, 3}, 0);
  cv::threshold(difference, mask, threshold, 255, cv::THRESH_BINARY);
  // Keep target/R segmentation unchanged. Lamp components get a separate
  // relative-color mask so bright, nearly white reflections do not enlarge
  // the arrow-chain envelope or masquerade as sparse active fan strips.
  cv::Mat arm_mask=mask.clone();
  if (config_.min_arm_color_ratio>0) {
    const auto& enemy=color==Color::Red ? red : blue;
    const auto& other=color==Color::Red ? blue : red;
    cv::Mat contrast,pure;
    cv::addWeighted(enemy,1-config_.min_arm_color_ratio,other,-1-config_.min_arm_color_ratio,
                    0,contrast,CV_32F);
    cv::GaussianBlur(contrast,contrast,{3,3},0);
    cv::compare(contrast,0,pure,cv::CMP_GT);
    cv::bitwise_and(arm_mask,pure,arm_mask);
  }
  if (config_.close_kernel > 1) {
    const auto kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               {config_.close_kernel,config_.close_kernel});
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                     kernel);
    cv::morphologyEx(arm_mask,arm_mask,cv::MORPH_CLOSE,kernel);
  }
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
  d.contour_count = contours.size();
  std::vector<Feature> features, hubs;
  std::vector<Target> targets;
  // Root shapes only. Child contours describe holes, never independent targets.
  std::vector<std::pair<double, int>> roots;
  for (std::size_t i = 0; i < contours.size(); ++i)
    if (hierarchy[i][3] < 0 && contours[i].size() >= 6) {
      const double area = cv::contourArea(contours[i]);
      if (area >= 15 && area <= config_.max_target_area) roots.emplace_back(area, int(i));
    }
  std::stable_sort(roots.begin(), roots.end(), [](auto a, auto b) { return a.first > b.first; });
  if (roots.size() > std::size_t(config_.max_features)) roots.resize(config_.max_features);
  for (const auto& [area, index] : roots) {
    const auto ellipse = cv::fitEllipse(contours[index]);
    const double minor = std::min(ellipse.size.width, ellipse.size.height);
    const double major = std::max(ellipse.size.width, ellipse.size.height);
    if (!inImage(ellipse.center, bgr.size()) || !std::isfinite(major) || minor < 2) continue;
    const auto moments = cv::moments(contours[index]);
    if (moments.m00 <= 0) continue;
    const Feature f{index, area, ellipse.center, ellipse};
    int holes = 0, gaps = 0;
    double hole_area = 0;
    bool concentric = false;
    std::vector<int> hole_indices;
    for (int child = hierarchy[index][2]; child >= 0; child = hierarchy[child][0]) {
      const double a = cv::contourArea(contours[child]);
      if (a < 5) continue;
      hole_indices.push_back(child);
      ++holes; hole_area += a;
      if (contours[child].size() < 6) continue;
      const auto e = cv::fitEllipse(contours[child]);
      const auto position = normalized(ellipse, e.center - ellipse.center);
      const double r = cv::norm(position);
      const double aspect = std::max(e.size.width, e.size.height) /
          std::max(1.0F, std::min(e.size.width, e.size.height));
      if (a / area >= 0.025 && a / area <= 0.20 && r >= 0.5 && r <= 0.9 && aspect >= 1.55)
        ++gaps;
      concentric |= r < 0.15 && a / area > 0.35;
    }
    const double fill = area / (CV_PI * minor * major / 4);
    double ellipse_residual = 0;
    for (const auto& point : contours[index])
      ellipse_residual += std::abs(cv::norm(normalized(ellipse, cv::Point2f(point) - ellipse.center)) - 1);
    ellipse_residual /= contours[index].size();
    const bool inactive = holes >= 4 && gaps >= 1 && gaps <= 4;
    const bool active = holes > 0 && (concentric || hole_area / area <= 0.30);
    if (area >= config_.min_target_area && major / minor <= config_.max_target_axis_ratio &&
        fill >= config_.min_ellipse_fill && fill <= 1.2 &&
        ellipse_residual <= config_.max_ellipse_residual && (inactive || active)) {
      targets.push_back({f, inactive, concentric, std::move(hole_indices)});
      continue;
    }
    const double perimeter = cv::arcLength(contours[index], true);
    std::vector<cv::Point> hull; cv::convexHull(contours[index], hull);
    const double hull_area = cv::contourArea(hull);
    const double rect_fill = area / std::max(1.0F, cv::minAreaRect(contours[index]).size.area());
    const auto bounds = cv::boundingRect(contours[index]);
    const bool clipped = bounds.x <= 1 || bounds.y <= 1 ||
        bounds.br().x >= bgr.cols - 1 || bounds.br().y >= bgr.rows - 1;
    if (area >= config_.min_center_area && area <= config_.max_center_area && major / minor <= 2.5 &&
        !clipped && rect_fill >= config_.min_center_rect_fill &&
        hole_area / area <= 0.5 && hull_area > 0 && area / hull_area >= 0.3 &&
        perimeter > 0 && 4 * CV_PI * area / (perimeter * perimeter) >= 0.1) {
      auto center = f;
      center.center = {float(moments.m10 / moments.m00), float(moments.m01 / moments.m00)};
      hubs.push_back(center);
    }
  }
  std::vector<std::vector<cv::Point>> arm_contours;
  cv::findContours(arm_mask,arm_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
  std::vector<std::pair<double,int>> arm_candidates;
  for (std::size_t i=0;i<arm_contours.size();++i) {
    const double area=cv::contourArea(arm_contours[i]);
    if (area>=15 && area<=config_.max_target_area) arm_candidates.emplace_back(area,int(i));
  }
  std::stable_sort(arm_candidates.begin(),arm_candidates.end(),[](auto a,auto b) { return a.first>b.first; });
  if (arm_candidates.size()>std::size_t(config_.max_features)) arm_candidates.resize(config_.max_features);
  for (const auto& [area,index] : arm_candidates) {
    const auto moments=cv::moments(arm_contours[index]);
    if (moments.m00<=0) continue;
    // An ellipse fitted to a concave arrow chain can have its center far
    // outside the lamp. Use measured moments for component association.
    const cv::Point2f center(float(moments.m10/moments.m00),float(moments.m01/moments.m00));
    if (inImage(center,bgr.size())) features.push_back({index,area,center,{}});
  }
  d.target_candidates = targets.size(); d.center_candidates = hubs.size();
  std::vector<Match> best;
  double best_score = -1;
  for (const auto& hub : hubs) {
    std::vector<Match> matches;
    for (std::size_t i = 0; i < targets.size(); ++i) {
      if (const auto match = matchTarget(i, targets[i], hub, features, contours,arm_contours,
                                        config_, bgr.size(), color)) matches.push_back(*match);
      else ++d.geometry_rejected;
    }
    std::stable_sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
      return a.quality > b.quality;
    });
    std::vector<Match> distinct;
    double score = 0;
    for (auto& m : matches) {
      const auto v = *m.keypoints.armorCenter() - hub.center;
      const bool duplicate = std::any_of(distinct.begin(), distinct.end(), [&](const auto& old) {
        const auto w = *old.keypoints.armorCenter() - hub.center;
        return std::abs(std::atan2(v.cross(w), v.dot(w))) < 0.6;
      });
      if (duplicate) continue;
      score += 1 + m.quality;
      distinct.push_back(std::move(m));
      if (distinct.size() == 5) break;
    }
    if (score > best_score) { best_score = score; best = std::move(distinct); }
  }
  std::vector<Keypoints> result;
  for (auto& m : best) {
    const auto& target = targets[m.target];
    std::vector<std::vector<cv::Point>> holes, fan_contours;
    for (const auto i : target.holes) holes.push_back(contours[i]);
    for (const auto i : m.fan_indices) fan_contours.push_back(arm_contours[i]);
    auto measured = std::make_shared<DetectedContours>(*m.keypoints.image_contours);
    extractScutFeaturePoints(*measured, target.feature.ellipse, m.keypoints.center_r,
                             m.keypoints.class_id == 0, holes, fan_contours);
    for (const auto& point : measured->gap_points) d.scut_gap_points += point.has_value();
    d.scut_fan_points += measured->fan_points.size();
    m.keypoints.image_contours = std::move(measured);
    result.push_back(std::move(m.keypoints));
  }
  d.detections = result.size(); d.arm_matches = result.size();
  d.target_ring_refined = result.size();
  if (!result.empty()) d.best_valid_keypoints = 5;
  diagnostics_ = d;
  (void)mode; // Target topology denotes activation in both modes; no model classes to filter.
  return result;
}
} // namespace Rune
