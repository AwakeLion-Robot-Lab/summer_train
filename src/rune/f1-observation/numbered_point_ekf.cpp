#include "rune/f1-observation/numbered_point_ekf.hpp"

#include <algorithm>
#include <cmath>

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
  const BladeBasis basis{Point(*center),radial,scale};

  std::array<std::optional<Point>,gaps+fan> observations;
  for (std::size_t i=0; i<gaps; ++i)
    if (output.gap_points[i] && inside(*output.gap_points[i],image_size))
      observations[i]=basis.local(*output.gap_points[i]);
  for (std::size_t i=0; i<output.fan_points.size(); ++i) {
    const int id=output.fanPointId(i);
    if (id>=0 && std::size_t(id)<fan && output.fanPointValid(i) &&
        inside(output.fan_points[i],image_size))
      observations[gaps+id]=basis.local(output.fan_points[i]);
  }

  const double pose_sigma=std::clamp(.025*diameter,1.0,3.0)/scale;
  const auto noise = [&](std::size_t i) {
    const double sigma=(i<gaps ? std::max(1.0,.02*diameter) : std::max(1.4,.028*diameter))/scale;
    return sigma*sigma;
  };
  const double max_innovation=std::max(3.0,.09*diameter)/scale;
  const auto inlier = [&](Point innovation, double variance) {
    const double error=innovation.dot(innovation);
    return error<=max_innovation*max_innovation && error<=9.21*variance;
  };

  PoseUpdate pose(pose_sigma,.025);
  const double process=std::clamp(dt/time_constant_s,0.0,1.0);
  std::array<double,gaps+fan> prior_variance{};
  struct Support { std::size_t index; Point innovation; double variance, score; };
  std::array<Support,gaps+fan> support;
  std::size_t count=0;
  for (std::size_t i=0; i<points_.size(); ++i) {
    const auto& point=points_[i];
    if (point.observations==0) continue;
    const double age=std::chrono::duration<double>(now-point.measured_at).count();
    if (age>max_gap_s) continue;
    prior_variance[i] = point.variance+noise(i)*process*process;
    if (point.observations<2 || !observations[i]) continue;
    const auto innovation=*observations[i]-point.position;
    const double variance=prior_variance[i]+noise(i);
    if (inlier(innovation,variance+pose_sigma*pose_sigma)) {
      const double error_variance=variance+point.innovation_variance;
      support[count++]={i,innovation,error_variance,error_variance+innovation.dot(innovation)};
    }
  }
  std::sort(support.begin(),support.begin()+count,[](const Support& a,const Support& b) {
    return a.score<b.score;
  });
  // 先选最稳的点，再保证至少有一个支撑点与它有足够间距。
  if (count>1) {
    const auto first=points_[support[0].index].position;
    const auto far=std::find_if(support.begin()+1,support.begin()+count,[&](const Support& s) {
      return cv::norm(points_[s.index].position-first)>=.08;
    });
    if (far!=support.begin()+count) std::rotate(support.begin()+1,far,far+1);
  }
  for (std::size_t j=0; j<std::min<std::size_t>(count,4); ++j) {
    const auto& s=support[j];
    pose.add(points_[s.index].position,s.innovation,s.variance);
  }
  // R 标短时不可用时，由至少两个有间距的已确认编号点校验叶片位置。
  if (!hub_measured && !pose.spread) return;
  const auto correction=pose.solve(.04*diameter/scale);
  const Point translation(correction[0],correction[1]);
  const double rotation=correction[2];

  for (std::size_t i=0; i<points_.size(); ++i) {
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
        if (inlier(residual,point.variance+variance)) {
          const double gain=point.variance/(point.variance+variance);
          point.position += gain*residual;
          point.variance *= 1-gain;
          point.innovation_variance=.8*point.innovation_variance+.2*residual.dot(residual);
          point.measured_at=now;
          point.observations=2;
        }
      }
      continue;
    }
    if (point.observations<2) continue;
    const auto projected=basis.image(rotate(point.position,rotation)+translation);
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
