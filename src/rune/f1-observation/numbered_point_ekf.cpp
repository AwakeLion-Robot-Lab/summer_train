#include "rune/f1-observation/numbered_point_ekf.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace Rune {
namespace {
using Point = cv::Point2d;

bool finite(Point p) { return std::isfinite(p.x) && std::isfinite(p.y); }
bool inside(Point p, cv::Size size) {
  return finite(p) && p.x>=0 && p.y>=0 && p.x<size.width && p.y<size.height;
}

struct BladeBasis {
  Point center, radial;
  double scale;

  Point local(Point p) const {
    p -= center;
    return {radial.dot(p)/scale, radial.cross(p)/scale};
  }

  Point image(Point p) const { return center + scale*(radial*p.x + Point(-radial.y,radial.x)*p.y); }
};

// 靶面椭圆归一化坐标系：消除大视角倾斜旋转时的周期性透视拉扁失真。
struct TargetEllipseBasis {
  Point center{};
  double a{1.0}, b{1.0};
  double ca{1.0}, sa{0.0};
  Point out_n{0.0, -1.0};
  Point right_n{1.0, 0.0};
  double minor_px{1.0};
  bool valid{false};

  static TargetEllipseBasis create(const cv::RotatedRect& ellipse, Point outward) {
    TargetEllipseBasis basis;
    basis.a = 0.5 * double(ellipse.size.width);
    basis.b = 0.5 * double(ellipse.size.height);
    basis.minor_px = std::min(basis.a, basis.b);
    if (!std::isfinite(basis.a) || !std::isfinite(basis.b) || basis.minor_px < 4.0) return basis;
    basis.center = Point(ellipse.center);
    const double rad = double(ellipse.angle) * std::numbers::pi / 180.0;
    basis.ca = std::cos(rad);
    basis.sa = std::sin(rad);
    const Point raw_out((outward.x * basis.ca + outward.y * basis.sa) / basis.a,
                        (-outward.x * basis.sa + outward.y * basis.ca) / basis.b);
    const double len = cv::norm(raw_out);
    if (!std::isfinite(len) || len < 1e-6) return basis;
    basis.out_n = raw_out * (1.0 / len);
    basis.right_n = {-basis.out_n.y, basis.out_n.x};
    basis.valid = true;
    return basis;
  }

  Point local(Point p) const {
    const Point d = p - center;
    const Point n((d.x * ca + d.y * sa) / a, (-d.x * sa + d.y * ca) / b);
    // 与 BladeBasis 保持统一轴向约定：x 沿径向向外，y 沿右侧切向。
    return {n.dot(out_n), n.dot(right_n)};
  }

  Point image(Point p) const {
    const Point n = p.x * out_n + p.y * right_n;
    const double lx = n.x * a;
    const double ly = n.y * b;
    return {center.x + ca * lx - sa * ly, center.y + sa * lx + ca * ly};
  }
};

Point rotate(Point p, double angle) {
  const double c=std::cos(angle), s=std::sin(angle);
  return {c*p.x-s*p.y,s*p.x+c*p.y};
}

// 在当前叶片坐标系内，对微小平移和转角做一次批量 EKF 更新。
// h(t,theta) = t + R(theta)*point，在线性化点 (0,0,0) 求雅可比矩阵。
struct PoseUpdate {
  cv::Matx33d information;
  cv::Vec3d residual{};
  std::optional<Point> first_point;
  bool spread{false};

  PoseUpdate(double translation_sigma, double rotation_sigma)
      : information(cv::Matx33d::diag({1/(translation_sigma*translation_sigma),
                                       1/(translation_sigma*translation_sigma),
                                       1/(rotation_sigma*rotation_sigma)})) {}

  void add(Point point, Point innovation, double variance) {
    const cv::Matx<double,2,3> h(1,0,-point.y,0,1,point.x);
    information += (h.t()*h)*(1/variance);
    residual += h.t()*cv::Vec2d(innovation.x,innovation.y)*(1/variance);
    if (first_point) spread |= cv::norm(point-*first_point) >= .08;
    else first_point = point;
  }

  cv::Vec3d solve(double shift_limit) const {
    if (!spread) return {};
    cv::Vec3d result;
    if (!cv::solve(information,residual,result,cv::DECOMP_CHOLESKY)) return {};
    const double shift=std::hypot(result[0],result[1]);
    if (shift>shift_limit) {
      result[0]*=shift_limit/shift;
      result[1]*=shift_limit/shift;
    }
    result[2]=std::clamp(result[2],-.04,.04);
    return result;
  }
};
}  // namespace

void NumberedPointEkf::update(const Keypoints& blade, DetectedContours& output, cv::Size image_size,
    TimePoint now, double max_gap_s, double time_constant_s) {
  const auto center = blade.armorCenter();
  if (!center || !inside(*center,image_size) || !output.target_ellipse ||
      blade.target_ring_status == TargetRingStatus::Rejected) return;

  const auto& ellipse = *output.target_ellipse;
  const double diameter=std::max(ellipse.size.width,ellipse.size.height);
  if (!std::isfinite(diameter) || diameter<4) return;
  const double dt=std::chrono::duration<double>(now-previous_time_).count();
  if (previous_time_==TimePoint{} || dt<=0 || dt>max_gap_s) *this={};

  const bool hub_measured = blade.point_valid[4] && inside(blade.center_r,image_size);
  Point radial;
  double scale;
  if (hub_measured) {
    radial=Point(*center)-Point(blade.center_r);
    scale=cv::norm(radial);
  } else {
    const double hub_age=std::chrono::duration<double>(now-hub_measured_at_).count();
    if (previous_scale_<=0 || hub_age>max_gap_s || !blade.point_valid[0] || !blade.point_valid[3]) return;
    radial=Point(blade.top)-Point(blade.bottom);
    scale=previous_scale_*diameter/previous_diameter_;
  }
  const double radial_length=cv::norm(radial);
  if (!std::isfinite(scale) || scale<5 || !std::isfinite(radial_length) || radial_length<2)
    return;
  radial *= 1/radial_length;
  if (previous_scale_>0 && (radial.dot(previous_radial_)<.91 ||
      scale<.8*previous_scale_ || scale>1.25*previous_scale_)) {
    *this={};
    if (!hub_measured) return;
  }
  const BladeBasis fan_basis{Point(*center),radial,scale};
  const auto gap_basis = TargetEllipseBasis::create(ellipse, radial);
  const double gap_scale = gap_basis.valid ? gap_basis.minor_px : scale;

  const auto toLocal = [&](std::size_t i, Point p) {
    return (i < gaps && gap_basis.valid) ? gap_basis.local(p) : fan_basis.local(p);
  };
  const auto toImage = [&](std::size_t i, Point p) {
    return (i < gaps && gap_basis.valid) ? gap_basis.image(p) : fan_basis.image(p);
  };
  const auto basisScale = [&](std::size_t i) {
    return (i < gaps && gap_basis.valid) ? gap_scale : scale;
  };

  std::array<std::optional<Point>,gaps+fan> observations;
  for (std::size_t i=0; i<gaps; ++i)
    if (output.gap_points[i] && inside(*output.gap_points[i],image_size))
      observations[i]=toLocal(i, *output.gap_points[i]);
  for (std::size_t i=0; i<output.fan_points.size(); ++i) {
    const int id=output.fanPointId(i);
    if (id>=0 && std::size_t(id)<fan && output.fanPointValid(i) &&
        inside(output.fan_points[i],image_size))
      observations[gaps+id]=fan_basis.local(output.fan_points[i]);
  }

  const auto noise = [&](std::size_t i) {
    if (i < gaps) {
      const double px_var = output.gap_point_variances[i] > 0.0F
          ? double(output.gap_point_variances[i])
          : std::pow(std::max(1.0, .02 * diameter), 2);
      return px_var / (gap_scale * gap_scale);
    }
    const double sigma = std::max(1.4, .028 * diameter) / scale;
    return sigma * sigma;
  };
  const auto poseSigmaFor = [&](std::size_t i) {
    return std::clamp(.025 * diameter, 1.0, 3.0) / basisScale(i);
  };
  const auto maxInnovationFor = [&](std::size_t i) {
    return std::max(3.0, .09 * diameter) / basisScale(i);
  };
  const auto inlier = [&](std::size_t i, Point innovation, double variance) {
    const double error = innovation.dot(innovation);
    const double max_innov = maxInnovationFor(i);
    return error <= max_innov * max_innov && error <= 9.21 * variance;
  };

  const double process=std::clamp(dt/time_constant_s,0.0,1.0);
  std::array<double,gaps+fan> prior_variance{};

  // 分别在靶面规范化坐标系（8个缺口点）和叶片相似坐标系（灯臂点）求解刚体微调。
  const auto solveGroupPose = [&](std::size_t begin, std::size_t end,
                                  double pose_sigma, double shift_limit,
                                  bool& has_spread) -> cv::Vec3d {
    PoseUpdate pose(pose_sigma, .025);
    struct Support { std::size_t index; Point innovation; double variance, score; };
    std::array<Support, gaps+fan> support;
    std::size_t count = 0;
    for (std::size_t i = begin; i < end; ++i) {
      const auto& point = points_[i];
      if (point.observations == 0) continue;
      const double age = std::chrono::duration<double>(now - point.measured_at).count();
      if (age > max_gap_s) continue;
      prior_variance[i] = point.variance + noise(i) * process * process;
      if (point.observations < 2 || !observations[i]) continue;
      const auto innovation = *observations[i] - point.position;
      const double variance = prior_variance[i] + noise(i);
      if (inlier(i, innovation, variance + pose_sigma * pose_sigma)) {
        const double error_variance = variance + point.innovation_variance;
        support[count++] = {i, innovation, error_variance, error_variance + innovation.dot(innovation)};
      }
    }
    std::sort(support.begin(), support.begin() + count, [](const Support& a, const Support& b) {
      return a.score < b.score;
    });
    if (count > 1) {
      const auto first = points_[support[0].index].position;
      const auto far = std::find_if(support.begin() + 1, support.begin() + count, [&](const Support& s) {
        return cv::norm(points_[s.index].position - first) >= .08;
      });
      if (far != support.begin() + count) std::rotate(support.begin() + 1, far, far + 1);
    }
    const std::size_t limit = (begin == 0) ? 8 : 4;
    for (std::size_t j = 0; j < std::min<std::size_t>(count, limit); ++j) {
      const auto& s = support[j];
      pose.add(points_[s.index].position, s.innovation, s.variance);
    }
    has_spread = pose.spread;
    return pose.solve(shift_limit);
  };

  bool gap_spread = false, fan_spread = false;
  const auto gap_correction = solveGroupPose(
      0, gaps, poseSigmaFor(0), .04 * diameter / gap_scale, gap_spread);
  const auto fan_correction = solveGroupPose(
      gaps, gaps + fan, poseSigmaFor(gaps), .04 * diameter / scale, fan_spread);

  // R 标短时不可用时，由至少两个有间距的已确认编号点校验叶片位置。
  if (!hub_measured && !gap_spread && !fan_spread) return;

  for (std::size_t i=0; i<points_.size(); ++i) {
    const auto& correction = (i < gaps) ? gap_correction : fan_correction;
    const Point translation(correction[0], correction[1]);
    const double rotation = correction[2];
    const double pose_sigma = poseSigmaFor(i);

    auto& point=points_[i];
    if (prior_variance[i]>0) point.variance=prior_variance[i];
    else point={};
    const double variance=noise(i)+pose_sigma*pose_sigma;
    if (observations[i]) {
      const Point local=rotate(*observations[i]-translation,-rotation);
      if (point.observations==0) {
        point={local,variance,0,now,1};
      } else {
        const auto residual=local-point.position;
        if (inlier(i, residual, point.variance+variance)) {
          const double gain=point.variance/(point.variance+variance);
          point.position += gain*residual;
          point.variance *= 1-gain;
          point.innovation_variance=.8*point.innovation_variance+.2*residual.dot(residual);
          point.measured_at=now;
          point.observations=2;
          // 将通过卡方校验且具备当帧图像实测的靶面关键点与后验方差回写至下游观测。
          if (i < gaps) {
            const auto filtered = toImage(i, rotate(point.position, rotation) + translation);
            if (inside(filtered, image_size)) {
              output.gap_points[i] = cv::Point2f(filtered);
              const double posterior_px_var = std::clamp(
                  (point.variance + 0.25 * pose_sigma * pose_sigma) * gap_scale * gap_scale,
                  0.36, 25.0);
              output.gap_point_variances[i] = float(posterior_px_var);
            }
          }
        }
      }
      continue;
    }
    if (point.observations<2) continue;
    const auto projected=toImage(i, rotate(point.position,rotation)+translation);
    if (!inside(projected,image_size)) continue;
    if (i<gaps) output.gap_display_predictions[i]=cv::Point2f(projected);
    else if (!output.fan_display_predictions[i-gaps])
      output.fan_display_predictions[i-gaps]=cv::Point2f(projected);
  }
  previous_time_=now;
  if (hub_measured) hub_measured_at_=now;
  previous_radial_=radial;
  previous_scale_=scale;
  previous_diameter_=diameter;
}

}  // namespace Rune
