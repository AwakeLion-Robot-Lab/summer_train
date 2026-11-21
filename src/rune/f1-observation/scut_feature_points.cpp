// Algorithm reference: scutrobotlab/rm_vision_core, commit 5c8085ae4b5f1f8f75e7c7d157ea60395f118429.
// rune_target_inactive.cpp: normalized gaps; rune_fan_active.cpp and hump_*.cpp:
// smoothed contour directions, opposite strip edges, projected strip endpoints.
#include "rune/f1-observation/scut_feature_points.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

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
struct Tip { Point base, vertex, direction; int strip; };

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

std::vector<Tip> stripTips(const Contour& contour, const std::vector<Strip>& strips) {
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
    if (!duplicate) tips.push_back({base, vertex, d, int(i)});
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

std::optional<std::array<Tip, 3>> topTips(const std::vector<Tip>& tips, Point center, Point hub) {
  if (tips.size() > 64) return {};
  std::optional<std::array<Tip, 3>> best;
  double cost = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < tips.size(); ++i)
    for (std::size_t j = i + 1; j < tips.size(); ++j)
      for (std::size_t k = j + 1; k < tips.size(); ++k) {
        std::array<Tip, 3> t{tips[i], tips[j], tips[k]};
        const auto average = (t[0].vertex + t[1].vertex + t[2].vertex) / 3.0F;
        auto middle = std::min_element(t.begin(), t.end(), [&](const auto& a, const auto& b) {
          return cv::norm(a.vertex - average) < cv::norm(b.vertex - average);
        });
        std::iter_swap(middle, t.begin() + 1);
        if ((t[0].vertex - center).cross(t[2].vertex - center) <= 0) std::swap(t[0], t[2]);
        const double collinear = 180 - std::max({
            angle(t[1].vertex - t[0].vertex, t[2].vertex - t[0].vertex),
            angle(t[0].vertex - t[1].vertex, t[2].vertex - t[1].vertex),
            angle(t[0].vertex - t[2].vertex, t[1].vertex - t[2].vertex)});
        const double score = alignment(t);
        const double left = cv::norm(t[0].vertex - t[1].vertex), right = cv::norm(t[1].vertex - t[2].vertex);
        if (collinear > 25 || score > 20 || score >= cost || std::min(left, right) < 1 ||
            std::max(left, right) / std::min(left, right) > 1.5 ||
            angle(t[1].direction, t[1].vertex - center) > 15 ||
            angle(-t[1].direction, hub - t[1].vertex) > 45) continue;
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

std::vector<Point> activeFan(const std::vector<Contour>& contours, Point hub) {
  std::vector<Tip> all_tips;
  std::vector<Point> all_points;
  for (const auto& raw : contours) {
    if (raw.size() < 30 || cv::contourArea(raw) < 30) continue;
    all_points.insert(all_points.end(), raw.begin(), raw.end());
    Contour extended = raw;
    extended.insert(extended.end(), raw.begin(), raw.begin() + raw.size() / 3);
    const auto strips = stripEdges(extended);
    const auto tips = stripTips(extended, strips);
    all_tips.insert(all_tips.end(), tips.begin(), tips.end());
    const auto top = topTips(tips, cv::fitEllipse(raw).center, hub);
    if (!top) continue;
    const auto& t = *top;
    const auto inward = -t[1].direction;
    if (angle(inward, hub - t[1].vertex) > 45) continue;
    std::optional<Point> bottom;
    double distance = 0;
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const auto a = Point(raw[i]) - t[1].base, b = Point(raw[(i + 1) % raw.size()]) - t[1].base;
      const double ca = a.cross(inward), cb = b.cross(inward);
      if (ca * cb > 0 || std::abs(ca - cb) < 1e-6) continue;
      const auto crossing = a + (b - a) * float(ca / (ca - cb));
      if (crossing.dot(inward) > distance) { distance = crossing.dot(inward); bottom = t[1].base + crossing; }
    }
    if (!bottom || angle(*bottom - t[1].base, inward) > 20) continue;
    std::array<std::optional<Point>, 2> sides;
    for (std::size_t i = 0; i < strips.size(); ++i) {
      if (int(i) == t[0].strip || int(i) == t[1].strip || int(i) == t[2].strip) continue;
      const auto base = (strips[i].up.center + strips[i].down.center) * .5F;
      const auto d = direction(strips[i].up.angle);
      const int side = cv::norm(base - t[0].base) < cv::norm(base - t[2].base) ? 0 : 1;
      const auto& tip = side == 0 ? t[0] : t[2];
      const auto toward_top = tip.base - base;
      if (angle(tip.direction, d) > 90 || tip.direction.cross(toward_top) * toward_top.cross(d) < 0 ||
          angle(tip.direction, toward_top) > 45 || angle(toward_top, d) > 45) continue;
      const auto p = intersection(tip.base, tip.direction, base, d);
      if (!p || !cv::boundingRect(raw).contains(cv::Point(*p))) continue;
      if (!sides[side] || cv::norm(*p - t[1].base) < cv::norm(*sides[side] - t[1].base)) sides[side] = p;
    }
    if (sides[0] && sides[1])
      return {t[0].vertex, t[1].vertex, t[2].vertex, *sides[1], *bottom, *sides[0]};
    return {t[0].vertex, t[1].vertex, t[2].vertex, *bottom};
  }
  // Occluded/disconnected fan: only the three measured, aligned strip tips.
  // No hidden bottom or side points are inferred from a fixed-size model.
  if (all_points.size() >= 5) {
    const auto top = topTips(all_tips, cv::fitEllipse(all_points).center, hub);
    if (top && angle(-(*top)[1].direction, hub - (*top)[1].vertex) <= 45)
      return {(*top)[0].vertex, (*top)[1].vertex, (*top)[2].vertex};
  }
  return {};
}
} // namespace

void extractScutFeaturePoints(DetectedContours& measured,
    const cv::RotatedRect& ellipse, Point hub, bool inactive,
    const std::vector<Contour>& holes, const std::vector<Contour>& fan_contours) {
  measured.target_ellipse = ellipse;
  measured.gap_points = {};
  measured.fan_points.clear();
  measured.fan_point_ids.clear();
  measured.fan_point_valid.clear();
  if (ellipse.size.width <= 0 || ellipse.size.height <= 0) return;
  if (inactive) {
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
  } else {
    measured.fan_points = activeFan(fan_contours, hub);
    if (measured.fan_points.size() == 6) measured.fan_point_ids = {0, 1, 2, 3, 4, 5};
    else if (measured.fan_points.size() == 4) measured.fan_point_ids = {0, 1, 2, 4};
    else if (measured.fan_points.size() == 3) measured.fan_point_ids = {0, 1, 2};
  }
}
} // namespace Rune
