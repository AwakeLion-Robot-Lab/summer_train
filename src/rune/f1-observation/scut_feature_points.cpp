// Algorithm reference: scutrobotlab/rm_vision_core, commit 5c8085ae4b5f1f8f75e7c7d157ea60395f118429.
// rune_target_inactive.cpp: normalized gaps; rune_fan_active.cpp and hump_*.cpp:
// smoothed contour directions, opposite strip edges, projected strip endpoints.
#include "rune/f1-observation/scut_feature_points.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace Rune {
namespace {
using Point = cv::Point2f;
using Contour = std::vector<cv::Point>;

Point unit(Point p) {
  const double length = cv::norm(p);
  return length > 1e-6 ? p * float(1 / length) : Point{};
}
double angle(Point a, Point b) {
  const double length = cv::norm(a) * cv::norm(b);
  return length > 1e-6 ? std::acos(std::clamp(a.dot(b) / length, -1.0, 1.0)) * 180 / CV_PI : 180;
}
Point direction(double degrees) {
  const double a = degrees * CV_PI / 180;
  return {float(std::cos(a)), float(std::sin(a))};
}
double angleDifference(double a, double b) { return std::abs(std::remainder(a - b, 360.0)); }
Point normalize(const cv::RotatedRect& e, Point p) {
  p -= e.center;
  const double a = e.angle * CV_PI / 180;
  return {float(2 * (p.x * std::cos(a) + p.y * std::sin(a)) / e.size.width),
          float(2 * (-p.x * std::sin(a) + p.y * std::cos(a)) / e.size.height)};
}
Point denormalize(const cv::RotatedRect& e, Point p) {
  const double a = e.angle * CV_PI / 180;
  const double x = p.x * e.size.width / 2, y = p.y * e.size.height / 2;
  return e.center + Point(float(x * std::cos(a) - y * std::sin(a)),
                          float(x * std::sin(a) + y * std::cos(a)));
}

void targetGaps(DetectedContours& out, const cv::RotatedRect& ellipse,
                Point hub, const std::vector<Contour>& holes) {
  struct Gap { Point left, middle, right; double quality; };
  std::array<std::optional<Gap>, 4> gaps;
  const auto inward = unit(hub - ellipse.center);
  const double outer_area = out.armor.size() >= 3 ? cv::contourArea(out.armor) : 0;
  if (outer_area <= 0 || inward == Point{}) return;
  for (const auto& hole : holes) {
    if (hole.size() < 10 || cv::pointPolygonTest(hole, ellipse.center, false) >= 0) continue;
    const double fraction = cv::contourArea(hole) / outer_area;
    if (fraction < .025 || fraction > .20) continue;
    std::vector<Point> points;
    for (const auto& p : hole) points.push_back(normalize(ellipse, Point(p)));
    const auto fit = cv::fitEllipse(points);
    const double distance = cv::norm(fit.center);
    const double short_axis = std::min(fit.size.width, fit.size.height);
    if (short_axis <= 0 || distance < .5 || distance > .8) continue;
    const double aspect = std::max(fit.size.width, fit.size.height) / short_axis;
    if (aspect < 1.55 || aspect > 8) continue;
    Point left = points.front(), right = points.front();
    for (const auto& p : points) {
      if (left.cross(p) < 0) left = p;
      if (right.cross(p) > 0) right = p;
    }
    const double span = angle(left, right);
    if (span < 60 || span > 100) continue;
    left = unit(left) * .7037F;
    right = unit(right) * .7037F;
    const auto middle = unit(fit.center) * .7037F;
    if (angle(right - left, middle) < 70 || angle(right - left, middle) > 110) continue;
    const auto middle_pixel = denormalize(ellipse, middle);
    const auto offset = middle_pixel - ellipse.center;
    const bool upper = inward.dot(offset) < 0;
    const bool left_side = inward.cross(offset) > 0;
    const int slot = upper ? (left_side ? 0 : 1) : (left_side ? 3 : 2);
    const double quality = std::abs(span - 90);
    if (!gaps[slot] || quality < gaps[slot]->quality)
      gaps[slot] = Gap{denormalize(ellipse, left), middle_pixel,
                       denormalize(ellipse, right), quality};
  }
  for (std::size_t i = 0; i < gaps.size(); ++i) {
    if (gaps[i]) out.gap_points[2 * i] = gaps[i]->middle;
    const auto next = (i + 1) % gaps.size();
    if (gaps[i] && gaps[next])
      out.gap_points[2 * i + 1] = (gaps[i]->right + gaps[next]->left) * .5F;
  }
}

struct Line {
  int first, last;
  double angle;
  Point center;
  int length() const { return last - first + 1; }
};
struct Strip { Line up, down; };
struct Tip { Point base, vertex, direction; std::size_t contour; std::size_t strip; };

// 已激活灯臂的编号按 R 标至靶心的方向固定。
enum FanPointId { LeftTip, CenterTip, RightTip, RightJoint, Base, LeftJoint, FanPointCount };
using FanPoints = std::array<std::optional<Point>, FanPointCount>;

struct BladeAxes {
  Point hub, outward;
  double length;

  double radial(Point p) const { return (p - hub).dot(outward) / length; }
  double lateral(Point p) const { return outward.cross(p - hub) / length; }
};

std::vector<Strip> stripEdges(const Contour& contour) {
  cv::Mat coordinates(2, int(contour.size()), CV_32F);
  for (int i = 0; i < coordinates.cols; ++i) {
    coordinates.at<float>(0, i) = contour[i].x;
    coordinates.at<float>(1, i) = contour[i].y;
  }
  cv::Mat directions, angles(1, coordinates.cols, CV_32F), gradient;
  const cv::Mat difference_kernel = (cv::Mat_<float>(1, 2) << -1, 1);
  cv::filter2D(coordinates, directions, -1, difference_kernel, {0, 0});
  const int length = std::clamp(int(.04 * contour.size()) | 1, 17, 101);
  cv::filter2D(directions, directions, -1, cv::getGaussianKernel(length, 5, CV_32F).t());
  double last = 0, turns = 0;
  for (int i = 0; i < coordinates.cols; ++i) {
    const double a = std::atan2(directions.at<float>(1, i), directions.at<float>(0, i)) * 180 / CV_PI;
    if (a - last < -180) ++turns;
    if (a - last > 180) --turns;
    last = a;
    angles.at<float>(0, i) = float(a + 360 * turns);
  }
  const cv::Mat gradient_kernel = (cv::Mat_<float>(1, 3) << -1, 0, 1);
  cv::filter2D(angles, gradient, -1, gradient_kernel);
  std::vector<Line> lines;
  for (int i = 0; i < angles.cols;) {
    if (std::abs(gradient.at<float>(0, i)) > 3) { ++i; continue; }
    const int first = i++;
    while (i < angles.cols && std::abs(gradient.at<float>(0, i)) <= 3) ++i;
    lines.push_back({first, i - 1, 0, {}});
  }
  for (std::size_t i = 1; i < lines.size();) {
    auto& before = lines[i - 1];
    const auto& after = lines[i];
    if (after.first - before.last < 20 &&
        std::abs(angles.at<float>(0, after.last) - angles.at<float>(0, before.first)) < 5) {
      before.last = after.last;
      lines.erase(lines.begin() + i);
    } else ++i;
  }
  // Bound work on noisy LED fragments before the pair/triple search.
  if (lines.size() > 128) return {};
  for (auto& line : lines) {
    for (int i = line.first; i <= line.last; ++i) {
      line.angle += angles.at<float>(0, i);
      line.center += Point(contour[i]);
    }
    line.angle /= line.length(); line.center *= 1.0F / line.length();
  }
  const auto rect = cv::minAreaRect(contour);
  const double max_distance = std::max(rect.size.width, rect.size.height) * .5;
  const auto distance = [](const Line& a, const Line& b) {
    const double average = (a.angle * a.length() + b.angle * b.length()) / (a.length() + b.length());
    return cv::norm(a.center - b.center) * std::abs(std::cos((average - a.angle) * CV_PI / 180));
  };
  std::vector<std::pair<int, int>> pairs;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    int best = -1; double nearest = max_distance;
    for (std::size_t j = i + 1; j < lines.size(); ++j) {
      if (std::abs(lines[i].angle - lines[j].angle) > 360 ||
          std::abs(lines[i].last - lines[j].first) > contour.size() / 3.0) break;
      if (std::abs(std::abs(lines[j].angle - lines[i].angle) - 180) > 20) continue;
      const double d = distance(lines[i], lines[j]);
      if (d <= nearest) { nearest = d; best = int(j); }
    }
    if (best >= 0) pairs.emplace_back(int(i), best);
  }
  // A reverse edge may belong to only one strip. Merge collinear fragments.
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    for (std::size_t j = i + 1; j < pairs.size();) {
      if (pairs[i].second != pairs[j].second) { ++j; continue; }
      auto& a = lines[pairs[i].first]; const auto& b = lines[pairs[j].first];
      if (distance(a, b) < 5) {
        a.angle = (a.angle * a.length() + b.angle * b.length()) / (a.length() + b.length());
        a.first = std::min(a.first, b.first); a.last = std::max(a.last, b.last);
      }
      pairs.erase(pairs.begin() + j);
    }
  }
  std::vector<Strip> strips;
  for (const auto& [a, b] : pairs) {
    if (direction(lines[a].angle).cross(lines[b].center - lines[a].center) <= 0)
      strips.push_back({lines[a], lines[b]});
  }
  const auto mergeAngle = [](Line& a, const Line& b) {
    if (double(std::max(a.length(), b.length())) / std::min(a.length(), b.length()) < 3) {
      const auto d = direction(a.angle) * float(a.length()) + direction(b.angle) * float(b.length());
      a.angle = std::atan2(d.y, d.x) * 180 / CV_PI;
    } else if (b.length() > a.length()) a.angle = b.angle;
  };
  for (std::size_t i = 0; i < strips.size(); ++i) {
    for (std::size_t j = i + 1; j < strips.size();) {
      auto& a = strips[i]; const auto& b = strips[j];
      if (cv::norm((a.up.center + a.down.center - b.up.center - b.down.center) * .5F) > 20 ||
          angleDifference(a.up.angle, b.up.angle) > 10 || angleDifference(a.down.angle, b.down.angle) > 10) {
        ++j; continue;
      }
      mergeAngle(a.up, b.up); mergeAngle(a.down, b.down);
      strips.erase(strips.begin() + j);
    }
  }
  for (auto& s : strips) {
    s.up.angle = std::remainder(s.up.angle, 360.0);
    s.down.angle = std::remainder(s.down.angle, 360.0);
    const double average = (s.up.angle + s.down.angle) * .5;
    s.up.angle = s.up.angle > average ? average + 90 : average - 90;
    s.down.angle = s.down.angle > average ? average + 90 : average - 90;
  }
  return strips;
}

std::vector<Tip> stripTips(const Contour& contour, const std::vector<Strip>& strips,
                          std::size_t contour_index) {
  std::vector<Tip> tips;
  for (std::size_t i = 0; i < strips.size(); ++i) {
    const auto& s = strips[i];
    const auto d = direction(s.up.angle);
    const auto base = (Point(contour[s.up.first]) + Point(contour[s.down.first])) * .5F;
    if (s.down.last <= s.up.first) continue;
    float furthest = -std::numeric_limits<float>::infinity();
    for (int j = s.up.first; j < s.down.last; ++j)
      furthest = std::max(furthest, (Point(contour[j]) - base).dot(d));
    const auto vertex = base + d * furthest;
    const bool duplicate = std::any_of(tips.begin(), tips.end(), [&](const auto& old) {
      return cv::norm(vertex - old.vertex) < (angle(d, old.direction) < 20 ? 10 : 20);
    });
    if (!duplicate) tips.push_back({base, vertex, d, contour_index, i});
  }
  return tips;
}

double alignment(const std::array<Tip, 3>& tips) {
  std::array<double, 3> a{};
  const double base = std::atan2(tips[0].direction.y, tips[0].direction.x);
  for (int i = 1; i < 3; ++i)
    a[i] = std::remainder(std::atan2(tips[i].direction.y, tips[i].direction.x) - base, 2 * CV_PI);
  const double average = (a[0] + a[1] + a[2]) / 3;
  double variance = 0;
  for (const auto value : a) variance += (value - average) * (value - average);
  return std::sqrt(variance / 3) * 180 / CV_PI;
}

std::optional<std::array<Tip, 3>> topTips(
    const std::vector<Tip>& tips, Point center, const BladeAxes& axes) {
  if (tips.size() > 64) return {};
  std::optional<std::array<Tip, 3>> best;
  double cost = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < tips.size(); ++i)
    for (std::size_t j = i + 1; j < tips.size(); ++j)
      for (std::size_t k = j + 1; k < tips.size(); ++k) {
        std::array<Tip, 3> t{tips[i], tips[j], tips[k]};
        std::sort(t.begin(), t.end(), [&](const auto& a, const auto& b) {
          return axes.lateral(a.vertex) < axes.lateral(b.vertex);
        });
        const double left_offset = axes.lateral(t[0].vertex);
        const double middle_offset = axes.lateral(t[1].vertex);
        const double right_offset = axes.lateral(t[2].vertex);
        const double spacing = std::min(middle_offset - left_offset, right_offset - middle_offset);
        if (spacing <= 0 || left_offset >= -.2 * spacing || right_offset <= .2 * spacing ||
            std::abs(middle_offset) > .35 * spacing)
          continue;
        const double collinear = 180 - std::max({
            angle(t[1].vertex - t[0].vertex, t[2].vertex - t[0].vertex),
            angle(t[0].vertex - t[1].vertex, t[2].vertex - t[1].vertex),
            angle(t[0].vertex - t[2].vertex, t[1].vertex - t[2].vertex)});
        const double score = alignment(t);
        const double left = cv::norm(t[0].vertex - t[1].vertex), right = cv::norm(t[1].vertex - t[2].vertex);
        if (collinear > 25 || score > 20 || score >= cost || std::min(left, right) < 1 ||
            std::max(left, right) / std::min(left, right) > 1.5 ||
            angle(t[1].direction, t[1].vertex - center) > 15 ||
            angle(t[1].direction, axes.outward) > 30 ||
            angle(-t[1].direction, axes.hub - t[1].vertex) > 45) continue;
        cv::Vec4f line;
        cv::fitLine(std::vector<Point>{t[0].vertex, t[1].vertex, t[2].vertex}, line, cv::DIST_L2, 0, .01, .01);
        const double perpendicular = angle({line[0], line[1]}, t[0].direction + t[1].direction + t[2].direction);
        if (std::min(perpendicular, 180 - perpendicular) < 45) continue;
        cost = score; best = t;
      }
  return best;
}

std::optional<Point> intersection(Point a, Point da, Point b, Point db) {
  const float determinant = da.cross(db);
  if (std::abs(determinant) < 1e-5) return {};
  const auto point = a + da * ((b - a).cross(db) / determinant);
  return std::isfinite(point.x) && std::isfinite(point.y) ? std::optional<Point>(point) : std::nullopt;
}

FanPoints tipsOnly(const std::array<Tip, 3>& tips) {
  FanPoints result;
  result[LeftTip] = tips[0].vertex;
  result[CenterTip] = tips[1].vertex;
  result[RightTip] = tips[2].vertex;
  return result;
}

FanPoints partialTips(const std::vector<Tip>& tips, const BladeAxes& axes) {
  std::array<const Tip*, 3> candidates{};
  std::array<bool, 3> ambiguous{};
  const double separation = std::max(3.0, .025 * axes.length);
  for (const auto& tip : tips) {
    const double radial = axes.radial(tip.vertex);
    if (radial < .42 || radial > .92 || angle(tip.direction, axes.outward) > 20 ||
        (tip.vertex - tip.base).dot(axes.outward) < .05 * axes.length)
      continue;
    const double lateral = axes.lateral(tip.vertex);
    int id = CenterTip;
    if (std::abs(lateral) <= .035 && std::abs(axes.lateral(tip.base)) <= .045) {
      id = CenterTip;
    } else if (std::abs(lateral) >= .07 && std::abs(lateral) <= .36 &&
               lateral * axes.lateral(tip.base) > 0) {
      id = lateral < 0 ? LeftTip : RightTip;
    } else {
      continue;
    }
    if (!candidates[id]) candidates[id] = &tip;
    else if (cv::norm(candidates[id]->vertex - tip.vertex) > separation) ambiguous[id] = true;
  }
  FanPoints result;
  const auto* middle = candidates[CenterTip];
  if (!middle || ambiguous[CenterTip]) return result;
  result[CenterTip] = middle->vertex;
  std::array<bool, 3> aligned{};
  for (const auto id : {LeftTip, RightTip}) {
    const auto* tip = candidates[id];
    if (tip && !ambiguous[id])
      aligned[id] = angle(tip->direction, middle->direction) <= 20 &&
          std::abs(axes.radial(tip->vertex) - axes.radial(middle->vertex)) <=
              .06 + .25 * std::abs(axes.lateral(tip->vertex));
  }
  // 若两侧均有候选但三端的联合几何检查失败，只保留已确认的中间灯条。
  if (aligned[LeftTip] != aligned[RightTip]) {
    const auto id = aligned[LeftTip] ? LeftTip : RightTip;
    result[id] = candidates[id]->vertex;
  }
  return result;
}

struct Joint {
  Point point, direction;
  double inward_reach;
};

FanPoints connectedFan(const Contour& raw, const Contour& extended,
                       const std::vector<Strip>& strips, const std::array<Tip, 3>& top,
                       const BladeAxes& axes, std::size_t contour_index) {
  auto result = tipsOnly(top);
  const auto bounds = cv::boundingRect(raw);
  const double span = cv::norm(top[2].vertex - top[0].vertex);
  std::array<std::optional<Joint>, 2> joints;
  for (std::size_t i = 0; i < strips.size(); ++i) {
    if (std::any_of(top.begin(), top.end(), [&](const auto& tip) {
          return tip.contour == contour_index && tip.strip == i;
        }))
      continue;
    const auto& strip = strips[i];
    const auto center = (strip.up.center + strip.down.center) * .5F;
    const auto direction_on_strip = direction(strip.up.angle);
    const double lateral = axes.lateral(center);
    if (std::abs(lateral) < .05 * span / axes.length) continue;
    const int side = lateral < 0 ? 0 : 1;
    const auto& tip = top[side == 0 ? 0 : 2];
    if (tip.contour != contour_index) continue;
    const auto toward_tip = tip.base - center;
    if (angle(tip.direction, direction_on_strip) > 90 ||
        tip.direction.cross(toward_tip) * toward_tip.cross(direction_on_strip) < 0 ||
        angle(tip.direction, toward_tip) > 45 || angle(toward_tip, direction_on_strip) > 45)
      continue;
    const auto point = intersection(tip.base, tip.direction, center, direction_on_strip);
    if (!point || !bounds.contains(cv::Point(*point)) ||
        cv::pointPolygonTest(raw, *point, true) < -2 ||
        axes.lateral(*point) * lateral <= 0 || axes.radial(*point) >= axes.radial(tip.vertex))
      continue;
    const auto inward = direction_on_strip.dot(axes.outward) > 0
        ? -direction_on_strip : direction_on_strip;
    double reach = 0;
    for (const auto& edge : {strip.up, strip.down})
      for (int j = edge.first; j <= edge.last; ++j)
        reach = std::max(reach, double((Point(extended[j]) - *point).dot(inward)));
    if (!joints[side] || cv::norm(*point - top[1].base) < cv::norm(joints[side]->point - top[1].base))
      joints[side] = Joint{*point, inward, reach};
  }
  if (joints[0]) result[LeftJoint] = joints[0]->point;
  if (joints[1]) result[RightJoint] = joints[1]->point;
  if (!joints[0] || !joints[1] || top[1].contour != contour_index) return result;

  // 中线与轮廓的交点可能是遮挡边缘。两侧实测灯条应同时延伸至汇合处。
  const auto& left = *joints[0];
  const auto& right = *joints[1];
  const auto convergence = intersection(left.point, left.direction, right.point, right.direction);
  const double tolerance = std::max(4.0, .16 * span);
  if (!convergence || std::abs(axes.lateral(*convergence)) * axes.length > tolerance ||
      axes.radial(*convergence) >= std::min(axes.radial(left.point), axes.radial(right.point)))
    return result;
  for (const auto& side : {left, right}) {
    const double distance = (*convergence - side.point).dot(side.direction);
    if (distance <= 0 || distance > side.inward_reach + tolerance) return result;
  }
  const auto inward = -top[1].direction;
  double furthest = 0;
  std::optional<Point> base;
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto a = Point(raw[i]) - top[1].base;
    const auto b = Point(raw[(i + 1) % raw.size()]) - top[1].base;
    const double ca = a.cross(inward), cb = b.cross(inward);
    if (ca * cb > 0 || std::abs(ca - cb) < 1e-6) continue;
    const auto crossing = a + (b - a) * float(ca / (ca - cb));
    const double distance = crossing.dot(inward);
    if (distance > furthest) { furthest = distance; base = top[1].base + crossing; }
  }
  if (base && cv::norm(*base - *convergence) <= tolerance) result[Base] = *base;
  return result;
}

FanPoints activeFan(const std::vector<Contour>& contours, const BladeAxes& axes) {
  struct MeasuredStrips {
    const Contour& raw;
    Contour extended;
    std::vector<Strip> strips;
  };
  std::vector<MeasuredStrips> measured_strips;
  std::vector<Tip> all_tips;
  std::vector<Point> all_points;
  FanPoints best;
  int best_count = 0;
  for (const auto& raw : contours) {
    if (raw.size() < 30 || cv::contourArea(raw) < 30) continue;
    all_points.insert(all_points.end(), raw.begin(), raw.end());
    Contour extended = raw;
    extended.insert(extended.end(), raw.begin(), raw.begin() + raw.size() / 3);
    auto strips = stripEdges(extended);
    const auto contour_index = measured_strips.size();
    const auto tips = stripTips(extended, strips, contour_index);
    all_tips.insert(all_tips.end(), tips.begin(), tips.end());
    const auto top = topTips(tips, cv::fitEllipse(raw).center, axes);
    if (top) {
      auto candidate = connectedFan(raw, extended, strips, *top, axes, contour_index);
      const int count = std::count_if(candidate.begin(), candidate.end(),
                                      [](const auto& point) { return point.has_value(); });
      if (count == FanPointCount) return candidate;
      if (count > best_count) { best = std::move(candidate); best_count = count; }
    }
    measured_strips.push_back({raw, std::move(extended), std::move(strips)});
  }
  if (best_count > 0) return best;
  // 遮挡将灯条分开时，先跨轮廓确认三个外端，再分别保留仍有实测依据的转折点。
  if (all_points.size() >= 5) {
    const auto top = topTips(all_tips, cv::fitEllipse(all_points).center, axes);
    if (top) {
      auto result = tipsOnly(*top);
      for (std::size_t i = 0; i < measured_strips.size(); ++i) {
        const auto& measured = measured_strips[i];
        const auto candidate = connectedFan(measured.raw, measured.extended, measured.strips, *top, axes, i);
        for (const auto id : {RightJoint, Base, LeftJoint})
          if (candidate[id]) result[id] = candidate[id];
      }
      return result;
    }
  }
  return partialTips(all_tips, axes);
}
} // namespace

void extractScutActiveFanPoints(DetectedContours& measured, Point hub, Point radial_reference,
                                const std::vector<Contour>& fan_contours) {
  measured.fan_points.clear();
  measured.fan_point_ids.clear();
  measured.fan_point_valid.clear();
  measured.fan_display_predictions={};
  const auto radial=radial_reference-hub;
  const double length=cv::norm(radial);
  if (length<2 || !std::isfinite(length)) return;
  const auto points=activeFan(fan_contours,{hub,unit(radial),length});
  for (int id=0; id<FanPointCount; ++id) {
    if (!points[id]) continue;
    measured.fan_points.push_back(*points[id]);
    measured.fan_point_ids.push_back(id);
    measured.fan_point_valid.push_back(true);
  }
}

void extractScutFeaturePoints(DetectedContours& measured,
    const cv::RotatedRect& ellipse, Point hub, bool inactive,
    const std::vector<Contour>& holes, const std::vector<Contour>& fan_contours) {
  measured.target_ellipse = ellipse;
  measured.gap_points = {};
  measured.gap_display_predictions = {};
  if (!inactive && ellipse.size.width>0 && ellipse.size.height>0) {
    extractScutActiveFanPoints(measured,hub,ellipse.center,fan_contours);
    return;
  }
  measured.fan_points.clear();
  measured.fan_point_ids.clear();
  measured.fan_point_valid.clear();
  measured.fan_display_predictions = {};
  if (ellipse.size.width <= 0 || ellipse.size.height <= 0) return;
  targetGaps(measured, ellipse, hub, holes);
  Contour points;
  for (const auto& c : fan_contours) points.insert(points.end(), c.begin(), c.end());
  if (points.size() < 3) return;
  const auto rect = cv::minAreaRect(points);
  const double width = std::min(rect.size.width, rect.size.height), height = std::max(rect.size.width, rect.size.height);
  if (width < 1 || height / width < 2 || height / width > 7) return;
  std::vector<cv::Point> hull;
  cv::convexHull(points, hull);
  if (cv::contourArea(hull) / rect.size.area() < .7) return;
  measured.fan_points.resize(4);
  rect.points(measured.fan_points.data());
  const auto inward = unit(hub - ellipse.center);
  const Point right{inward.y, -inward.x};
  // OpenCV may change the first corner when a rectangle crosses an axis.
  // Sort in the blade frame before assigning persistent semantic IDs.
  auto& p = measured.fan_points;
  std::sort(p.begin(), p.end(), [&](Point a, Point b) { return a.dot(inward) < b.dot(inward); });
  if (p[0].dot(right) > p[1].dot(right)) std::swap(p[0], p[1]);
  if (p[2].dot(right) < p[3].dot(right)) std::swap(p[2], p[3]);
  measured.fan_point_ids = {0, 1, 2, 3};
}
} // namespace Rune
