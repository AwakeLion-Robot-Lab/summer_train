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

bool touchesEdge(const cv::Rect& bounds, const cv::Rect& area) {
  return bounds.x <= area.x + 1 || bounds.y <= area.y + 1 ||
         bounds.br().x >= area.br().x - 1 || bounds.br().y >= area.br().y - 1;
}

// 在拟合出的环内限制暗区连通范围，保留与外部背景相通的窄缺口。
std::vector<std::vector<cv::Point>> originalGaps(const cv::Mat& binary,
    const cv::RotatedRect& ellipse, const cv::Rect& bounds, double area) {
  cv::Mat inside=cv::Mat::zeros(bounds.size(),CV_8U);
  auto inner=ellipse;
  inner.center-=cv::Point2f(bounds.tl());
  inner.size*=.94F;
  cv::ellipse(inside,inner,cv::Scalar(255),cv::FILLED);
  cv::Mat dark;
  cv::bitwise_not(binary(bounds),dark);
  cv::bitwise_and(dark,inside,dark);
  std::vector<std::vector<cv::Point>> candidates,result;
  cv::findContours(dark,candidates,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE,bounds.tl());
  for (auto& contour:candidates) {
    if (contour.size()<10) continue;
    const double fraction=cv::contourArea(contour)/area;
    if (fraction<.025 || fraction>.20) continue;
    const auto fit=cv::fitEllipse(contour);
    const auto p=normalized(ellipse,fit.center-ellipse.center);
    const double radius=cv::norm(p);
    if (radius<.5 || radius>.8 || std::min(fit.size.width,fit.size.height)<=0 ||
        std::max(fit.size.width,fit.size.height)<1.55*std::min(fit.size.width,fit.size.height))
      continue;
    const double angle=std::atan2(p.y,p.x);
    std::array<bool,2> bright{};
    for (int i=0;i<2;++i) for (const double offset:{.65,.85,1.05}) {
      const double theta=angle+(i==0 ? offset : -offset);
      const cv::Point pixel=ellipse.center+denormalized(ellipse,
          {float(radius*std::cos(theta)),float(radius*std::sin(theta))});
      if (bounds.contains(pixel) && binary.at<unsigned char>(pixel)) bright[i]=true;
    }
    if (bright[0] && bright[1]) result.push_back(std::move(contour));
  }
  return result;
}

struct Target {
  Feature feature;
  bool inactive;
  bool concentric;
  std::vector<int> holes;
  std::vector<std::vector<cv::Point>> original_gaps;
};
struct Match {
  const Target* target;
  std::array<cv::Point2f,4> anchors;
  bool inactive;
  double quality;
  int main_lamp;
  std::vector<int> fan_indices;
};

std::vector<RuneFanCorner> fanCorners(const DetectedContours& measured, bool activated,
                                     bool allow_partial=false) {
  std::vector<RuneFanCorner> corners;
  corners.reserve(measured.fan_points.size());
  for (std::size_t i=0; i<measured.fan_points.size(); ++i) {
    if (measured.fanPointValid(i))
      corners.push_back({measured.fanPointId(i),measured.fan_points[i]});
  }
  if (activated && corners.size()!=6) {
    if (corners.size()>=3 && corners[0].id==0 && corners[1].id==1 && corners[2].id==2)
      corners.resize(3);
    else if (allow_partial)
      std::erase_if(corners,[](const auto& corner) { return corner.id<0 || corner.id>2; });
    else
      corners.clear();
  }
  return corners;
}

std::vector<Keypoints> keypointsFromGroup(RuneGroup group) {
  std::vector<Keypoints> result;
  result.reserve(group.blades.size());
  for (auto& blade : group.blades) {
    auto measured = std::make_shared<DetectedContours>();
    measured->color = group.color;
    measured->center_r = group.center.contour;
    measured->arm = std::move(blade.fan.contour);
    measured->fan_points.reserve(blade.fan.corners.size());
    measured->fan_point_ids.reserve(blade.fan.corners.size());
    for (const auto& corner : blade.fan.corners) {
      measured->fan_points.push_back(corner.pixel);
      measured->fan_point_ids.push_back(corner.id);
    }

    Keypoints k;
    k.center_r = group.center.position;
    k.class_id = blade.fan.activated ? 1 : 0;
    k.confidence = blade.confidence;
    k.point_confidences.fill(k.confidence);
    if (auto& target = blade.target) {
      k.top = target->anchors[0]; k.left = target->anchors[1];
      k.right = target->anchors[2]; k.bottom = target->anchors[3];
      k.target_ring_status = TargetRingStatus::Verified;
      measured->target_ellipse = target->ellipse;
      measured->armor = std::move(target->contour);
      measured->gap_points = target->gap_points;
    } else {
      // 旧接口用四个坐标限定灯臂的感兴趣区域。
      k.top = k.left = cv::Point2f(blade.fan.bounds.tl());
      k.right = k.bottom = cv::Point2f(blade.fan.bounds.br()-cv::Point(1,1));
      k.point_valid = {false,false,false,false,true};
      k.point_confidences = {0,0,0,0,k.confidence};
    }
    if (group.center.contour.empty()) {
      k.point_valid[4] = false;
      k.point_confidences[4] = 0;
    }
    if (blade.fan.activated) measured->fan_point_valid.assign(blade.fan.corners.size(),true);
    k.image_contours = std::move(measured);
    result.push_back(std::move(k));
  }
  return result;
}

std::vector<Feature> inferCenters(const std::vector<Target>& targets,
    const std::vector<Feature>& lamps, const ScutContourConfig& config, cv::Size image_size) {
  struct Axis {
    std::size_t target;
    int lamp;
    cv::Point2f direction;
    double expected_distance;
  };
  struct Candidate {
    cv::Point2f center;
    double error;
  };
  std::vector<Axis> axes;
  // 按已有的面积排序限制候选数量，避免杂乱轮廓使两两求交开销激增。
  for (std::size_t i=0; i<std::min<std::size_t>(targets.size(),10); ++i) {
    const auto& ring=targets[i].feature;
    int count=0;
    for (const auto& lamp : lamps) {
      if (lamp.area>3*ring.area) continue;
      const auto delta=lamp.center-ring.center;
      const double radius=cv::norm(normalized(ring.ellipse,delta));
      const double fraction=radius*config.radius_ratio;
      if (radius<1.1 || fraction<.16 || fraction>.86) continue;
      const double length=cv::norm(delta);
      axes.push_back({i,lamp.index,delta*float(1/length),length/fraction});
      if (++count==10) break;
    }
  }

  std::vector<Candidate> candidates;
  for (std::size_t i=0; i<axes.size(); ++i) {
    const auto& a=axes[i];
    const auto& ring_a=targets[a.target].feature;
    for (std::size_t j=i+1; j<axes.size(); ++j) {
      const auto& b=axes[j];
      const double cross=a.direction.cross(b.direction);
      if (a.target==b.target || a.lamp==b.lamp || std::abs(cross)<.25) continue;
      const auto& ring_b=targets[b.target].feature;
      const auto offset=ring_b.center-ring_a.center;
      const double range_a=offset.cross(b.direction)/cross;
      const double range_b=offset.cross(a.direction)/cross;
      if (range_a<=0 || range_b<=0) continue;
      const double error_a=std::abs(a.expected_distance/range_a-1);
      const double error_b=std::abs(b.expected_distance/range_b-1);
      if (std::max(error_a,error_b)>config.radius_ratio_tolerance) continue;
      const auto center=ring_a.center+a.direction*float(range_a);
      if (!inImage(center,image_size)) continue;
      const auto radial_a=normalized(ring_a.ellipse,ring_a.center-center);
      const auto radial_b=normalized(ring_a.ellipse,ring_b.center-center);
      const double angle=std::atan2(std::abs(radial_a.cross(radial_b)),radial_a.dot(radial_b));
      if (angle<CV_PI/5 || std::abs(std::remainder(angle,2*CV_PI/5))>CV_PI/9) continue;
      candidates.push_back({center,error_a+error_b});
    }
  }
  std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b) {
    return a.error<b.error;
  });
  std::vector<Feature> centers;
  for (const auto& candidate : candidates) {
    if (std::any_of(centers.begin(),centers.end(),[&](const auto& center) {
      return cv::norm(center.center-candidate.center)<3;
    })) continue;
    centers.push_back({-1,0,candidate.center,{}});
    if (centers.size()==24) break;
  }
  return centers;
}

std::optional<Match> matchTarget(const Target& target,
    const Feature& hub, const std::vector<Feature>& features,
    const std::vector<std::vector<cv::Point>>& arm_contours,
    const ScutContourConfig& config, cv::Size image_size) {
  const auto& t = target.feature;
  const auto delta = t.center - hub.center;
  const double distance = cv::norm(delta);
  const bool measured_hub = hub.index>=0;
  if (distance < 5 || (measured_hub &&
      (hub.area > t.area * 0.5 || hub.area < t.area * config.min_center_target_area_ratio)))
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
    active_fan=!fanCorners(measured,true).empty();
  }
  if (inactive ? !inactive_arm : !active_fan) return {};
  if (!inImage(hub.center,image_size)) return {};
  const std::array anchors{t.center+vertical,t.center-horizontal,t.center+horizontal,t.center-vertical};
  for (const auto p : anchors)
    if (!inImage(p, image_size)) return {};
  const double quality = 1 - 0.5 * error / config.radius_ratio_tolerance -
      0.2 * lateral_sum / (pieces * config.max_arm_offset_ratio);
  return Match{&target,anchors,inactive,quality,arm_index,std::move(fan_indices)};
}

std::vector<RuneBladeFeature> matchLampsWithoutRings(const Feature& hub, const std::vector<Match>& paired,
    const std::vector<Feature>& lamps, const std::vector<cv::Point>& hub_contour,
    const std::vector<std::vector<cv::Point>>& lamp_contours, cv::Size image_size) {
  const auto hub_bounds=cv::boundingRect(hub_contour);
  const double hub_size=std::max(hub_bounds.width,hub_bounds.height);
  const cv::Rect image_bounds({0,0},image_size);
  struct Lamp {
    const Feature& feature;
    bool strip;
  };
  std::vector<Lamp> candidates;
  std::vector<bool> used(lamp_contours.size());
  for (const auto& match : paired)
    for (const auto index : match.fan_indices) used[index]=true;
  for (const auto& lamp : lamps) {
    const double distance=cv::norm(lamp.center-hub.center);
    if (used[lamp.index] || distance<hub_size || distance>14*hub_size || lamp.area<30) continue;
    const auto& contour=lamp_contours[lamp.index];
    const auto rect=cv::minAreaRect(contour);
    const double minor=std::min(rect.size.width,rect.size.height);
    const double major=std::max(rect.size.width,rect.size.height);
    if (minor<2 || touchesEdge(cv::boundingRect(contour),image_bounds)) continue;
    const double aspect=major/minor;
    const bool strip=aspect>=2;
    if (!strip && lamp.area>.75*rect.size.area()) continue;
    candidates.push_back({lamp,strip});
    if (candidates.size()==24) break;
  }

  struct MatchWithoutRing {
    RuneBladeFeature blade;
    cv::Point2f direction;
    std::vector<int> components;
  };
  std::vector<MatchWithoutRing> matches;
  std::vector<std::vector<int>> tested_groups;
  for (const auto& seed : candidates) {
    std::vector<int> group;
    const auto radial=seed.feature.center-hub.center;
    const double distance=cv::norm(radial);
    if (seed.strip) {
      const auto axis=radial*float(1/distance);
      for (const auto& lamp : candidates) {
        const auto delta=lamp.feature.center-hub.center;
        const double along=delta.dot(axis)/distance;
        if (lamp.strip && along>.60 && along<1.40 && std::abs(delta.cross(axis))<.62*distance)
          group.push_back(lamp.feature.index);
      }
      if (group.size()<3) continue;
    } else group.push_back(seed.feature.index);
    if (std::find(tested_groups.begin(),tested_groups.end(),group)!=tested_groups.end()) continue;
    tested_groups.push_back(group);

    cv::Rect bounds;
    cv::Point2f lamp_center;
    double total_area=0;
    std::vector<std::vector<cv::Point>> visible;
    for (const auto index : group) {
      const auto& contour=lamp_contours[index];
      const auto m=cv::moments(contour);
      lamp_center+=cv::Point2f(float(m.m10),float(m.m01));
      total_area+=m.m00;
      bounds|=cv::boundingRect(contour);
      visible.push_back(contour);
    }
    DetectedContours measured;
    extractScutActiveFanPoints(measured,hub.center,lamp_center*float(1/total_area),visible);
    auto corners=fanCorners(measured,true);
    if (corners.empty()) continue;
    const auto& tips=measured.fan_points;
    const auto direction=tips[1]-hub.center;
    const double length=cv::norm(direction);
    const double span=cv::norm(tips[2]-tips[0]);
    if (span<hub_size || length<.65*span || length>4.2*span || length<5) continue;
    // 实测转折点参与几何验证和置信度评估。
    // 缺少转折点时，要求两侧灯条延伸过中间灯条。
    const double outer_lead=((tips[0]+tips[2])*.5F-tips[1]).dot(direction)/length;
    if (tips.size()==3 && (outer_lead<.035*span || outer_lead>.28*span)) continue;
    const auto main=std::max_element(visible.begin(),visible.end(),[&](const auto& a,const auto& b) {
      return cv::pointPolygonTest(a,tips[1],true)<cv::pointPolygonTest(b,tips[1],true);
    });
    RuneFanFeature fan{true,bounds,std::move(*main),std::move(corners)};
    const float confidence=.70F+.05F*float(tips.size()-3);
    matches.push_back({{std::nullopt,std::move(fan),confidence},direction,std::move(group)});
  }

  std::stable_sort(matches.begin(),matches.end(),[](const auto& a,const auto& b) {
    return a.blade.confidence>b.blade.confidence;
  });
  std::vector<cv::Point2f> directions;
  for (const auto& match : paired) directions.push_back(match.target->feature.center-hub.center);
  std::vector<RuneBladeFeature> result;
  for (auto& match : matches) {
    if (directions.size()==5) break;
    if (std::any_of(directions.begin(),directions.end(),[&](const auto& direction) {
      return std::abs(std::atan2(direction.cross(match.direction),direction.dot(match.direction)))<.6;
    }) || std::any_of(match.components.begin(),match.components.end(),[&](const auto index) {
      return used[index];
    })) continue;
    for (const auto index : match.components) used[index]=true;
    directions.push_back(match.direction);
    result.push_back(std::move(match.blade));
  }
  return result;
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

ScutContourDetector::ScutContourDetector(ScutContourConfig config, bool gimbal_only)
    : config_(config), gimbal_only_(gimbal_only) {
  validate(config_);
}
std::vector<Keypoints> ScutContourDetector::detect(const cv::Mat& bgr) {
  return detect(bgr, Mode::Small, Color::Blue);
}
std::vector<Keypoints> ScutContourDetector::detect(const cv::Mat& bgr, Mode mode) {
  return detect(bgr, mode, Color::Blue);
}
std::vector<Keypoints> ScutContourDetector::detect(const cv::Mat& bgr, Mode mode, Color color) {
  auto group = detectGroup(bgr, mode, color);
  if (!group) return {};
  return keypointsFromGroup(std::move(*group));
}
std::optional<RuneGroup> ScutContourDetector::detectGroup(const cv::Mat& bgr, Mode mode, Color color) {
  diagnostics_.reset();
  if (bgr.empty() || bgr.type() != CV_8UC3)
    throw std::invalid_argument("SCUT rune detector requires a nonempty CV_8UC3 BGR image");
  cv::Mat red, blue, difference, arm_purity, closing_kernel;
  cv::extractChannel(bgr, red, 2);
  cv::extractChannel(bgr, blue, 0);
  const auto& enemy = color == Color::Red ? red : blue;
  const auto& other = color == Color::Red ? blue : red;
  cv::subtract(enemy, other, difference);
  cv::GaussianBlur(difference, difference, {3, 3}, 0);
  if (config_.min_arm_color_ratio > 0) {
    cv::Mat contrast;
    cv::addWeighted(enemy, 1-config_.min_arm_color_ratio, other, -1-config_.min_arm_color_ratio,
                    0, contrast, CV_32F);
    cv::GaussianBlur(contrast, contrast, {3, 3}, 0);
    cv::compare(contrast, 0, arm_purity, cv::CMP_GT);
  }
  if (config_.close_kernel > 1)
    closing_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               {config_.close_kernel, config_.close_kernel});
  const int threshold = color == Color::Red ? config_.red_threshold : config_.blue_threshold;
  auto result = detectAtThreshold(difference, arm_purity, closing_kernel, mode, color, threshold,
                                  gimbal_only_);
  if ((!result || result->center.contour.empty()) &&
      config_.fallback_threshold_offset > 0 && threshold < 254) {
    const auto first_diagnostics=diagnostics_;
    auto stricter = detectAtThreshold(difference, arm_purity, closing_kernel, mode, color,
                                      std::min(254, threshold + config_.fallback_threshold_offset),
                                      gimbal_only_ && !result);
    if (stricter || !result) result=std::move(stricter);
    else diagnostics_=first_diagnostics;
    diagnostics_->contour_passes = 2;
  }
  return result;
}
std::optional<RuneGroup> ScutContourDetector::detectAtThreshold(
    const cv::Mat& difference, const cv::Mat& arm_purity, const cv::Mat& closing_kernel,
    Mode mode, Color color, int threshold, bool recover_center) {
  const auto image_size = difference.size();
  const cv::Rect image_bounds({0, 0}, image_size);
  DetectorDiagnostics d;
  d.contour_passes = 1;
  cv::Mat binary, mask, arm_mask;
  cv::threshold(difference, binary, threshold, 255, cv::THRESH_BINARY);
  if (closing_kernel.empty()) mask=binary;
  else cv::morphologyEx(binary,mask,cv::MORPH_CLOSE,closing_kernel);
  // Keep target/R segmentation unchanged. Lamp components get a separate
  // relative-color mask so bright, nearly white reflections do not enlarge
  // the arrow-chain envelope or masquerade as sparse active fan strips.
  if (arm_purity.empty()) {
    arm_mask = mask.clone();
  } else {
    cv::bitwise_and(binary, arm_purity, arm_mask);
    if (!closing_kernel.empty())
      cv::morphologyEx(arm_mask, arm_mask, cv::MORPH_CLOSE, closing_kernel);
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
    const auto bounds = cv::boundingRect(contours[index]);
    const bool clipped = touchesEdge(bounds, image_bounds);
    const auto ellipse = cv::fitEllipse(contours[index]);
    const double minor = std::min(ellipse.size.width, ellipse.size.height);
    const double major = std::max(ellipse.size.width, ellipse.size.height);
    if (!inImage(ellipse.center, image_size) || !std::isfinite(major) || minor < 2) continue;
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
    bool inactive = holes >= 4 && gaps >= 1 && gaps <= 4;
    const bool active = holes > 0 && (concentric || hole_area / area <= 0.30);
    if (area >= config_.min_target_area && major / minor <= config_.max_target_axis_ratio &&
        fill >= config_.min_ellipse_fill && fill <= 1.2 &&
        ellipse_residual <= config_.max_ellipse_residual) {
      std::vector<std::vector<cv::Point>> recovered;
      if (!inactive && !concentric && !clipped && !closing_kernel.empty()) {
        recovered=originalGaps(binary,ellipse,bounds,area);
        inactive=recovered.size()>=3 && recovered.size()<=4;
        if (!inactive) recovered.clear();
      }
      if (inactive || active) {
        targets.push_back({f,inactive,concentric,std::move(hole_indices),std::move(recovered)});
        continue;
      }
    }
    const double perimeter = cv::arcLength(contours[index], true);
    std::vector<cv::Point> hull; cv::convexHull(contours[index], hull);
    const double hull_area = cv::contourArea(hull);
    const double rect_fill = area / std::max(1.0F, cv::minAreaRect(contours[index]).size.area());
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
  cv::findContours(arm_mask, arm_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
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
    if (inImage(center,image_size)) features.push_back({index,area,center,{}});
  }
  d.target_candidates = targets.size(); d.center_candidates = hubs.size();
  std::vector<Match> best;
  std::optional<Feature> best_hub;
  double best_score = -1;
  const auto score_center = [&](const Feature& hub) {
    const bool inferred=hub.index<0;
    std::vector<Match> matches;
    for (const auto& target : targets) {
      if (auto match = matchTarget(target,hub,features,arm_contours,config_,image_size))
        matches.push_back(std::move(*match));
      else ++d.geometry_rejected;
    }
    std::stable_sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
      return a.quality > b.quality;
    });
    std::vector<Match> distinct;
    double score = 0;
    for (auto& m : matches) {
      const auto v = m.target->feature.center - hub.center;
      const bool duplicate = std::any_of(distinct.begin(), distinct.end(), [&](const auto& old) {
        const auto w = old.target->feature.center - hub.center;
        return m.main_lamp==old.main_lamp ||
            std::abs(std::atan2(v.cross(w), v.dot(w))) < 0.6;
      });
      if (duplicate) continue;
      score += 1 + m.quality;
      distinct.push_back(std::move(m));
      if (distinct.size() == 5) break;
    }
    if (inferred && distinct.size()<2) return;
    if (score > best_score) {
      best_score = score;
      best = std::move(distinct);
      best_hub=hub;
    }
  };
  for (const auto& hub : hubs) score_center(hub);
  if (recover_center && best.empty() && targets.size()>=2) {
    for (const auto& hub : inferCenters(targets,features,config_,image_size)) score_center(hub);
  }
  std::vector<RuneBladeFeature> lamps_without_rings;
  if (gimbal_only_ && mode==Mode::Big) {
    const auto match_lamps=[&](const Feature& hub) {
      return matchLampsWithoutRings(hub,best,features,contours[hub.index],arm_contours,image_size);
    };
    if (!best.empty() && best_hub && best_hub->index>=0) {
      lamps_without_rings=match_lamps(*best_hub);
    } else if (best.empty()) {
      double best_lamp_score=0;
      for (std::size_t i=0; i<std::min<std::size_t>(hubs.size(),8); ++i) {
        auto matches=match_lamps(hubs[i]);
        double score=0;
        for (const auto& match : matches) score+=1+match.confidence;
        if (score>best_lamp_score) {
          best_lamp_score=score;
          best_hub=hubs[i];
          lamps_without_rings=std::move(matches);
        }
      }
    }
  }
  if (best.empty() && lamps_without_rings.empty()) {
    diagnostics_ = d;
    return std::nullopt;
  }
  const auto& hub = best_hub.value();
  RuneGroup group{color,{hub.center,{}},{}};
  if (hub.index>=0) group.center.contour = std::move(contours[hub.index]);
  group.blades.reserve(best.size()+lamps_without_rings.size());
  for (const auto& match : best) {
    const auto& target = *match.target;
    std::vector<std::vector<cv::Point>> holes, fan_contours;
    if (!target.original_gaps.empty()) holes=target.original_gaps;
    else for (const auto i : target.holes) holes.push_back(contours[i]);
    cv::Rect fan_bounds;
    for (const auto i : match.fan_indices) {
      const auto& contour = arm_contours[i];
      fan_bounds |= cv::boundingRect(contour);
      fan_contours.push_back(contour);
    }
    DetectedContours measured;
    measured.armor = contours[target.feature.index];
    extractScutFeaturePoints(measured,target.feature.ellipse,hub.center,match.inactive,holes,fan_contours);
    // 保留实测的主轮廓，分离的灯条只提供匹配和角点证据。
    const auto& main_lamp = arm_contours[match.main_lamp];
    RuneTargetFeature target_feature{target.feature.ellipse,std::move(measured.armor),
                                     match.anchors,measured.gap_points};
    // 靶环与灯臂已经匹配，可以保留独立确认了身份的部分外端。
    RuneFanFeature fan{!match.inactive,fan_bounds,main_lamp,
                       fanCorners(measured,!match.inactive,true)};
    for (const auto& point : target_feature.gap_points) d.scut_gap_points += point.has_value();
    d.scut_fan_points += fan.corners.size();
    group.blades.push_back({std::move(target_feature),std::move(fan),
                            float(std::clamp(match.quality,0.1,1.0))});
  }
  for (auto& blade : lamps_without_rings) {
    d.scut_fan_points += blade.fan.corners.size();
    group.blades.push_back(std::move(blade));
  }
  d.detections = group.blades.size(); d.arm_matches = group.blades.size();
  diagnostics_ = d;
  return group;
}
} // namespace Rune
