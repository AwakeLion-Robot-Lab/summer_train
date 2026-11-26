#include "inactive_arm_refiner.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace Rune::detail {
namespace {
using Corners = std::array<cv::Point2f,4>;

struct ArmAxes {
  cv::Point2f origin;
  cv::Point2f forward;
  float length;

  cv::Point2f local(cv::Point2f p) const {
    const auto delta=p-origin;
    return {delta.dot(forward),float(forward.cross(delta))};
  }

  cv::Point2f pixel(float along,float across) const {
    return origin+along*forward+across*cv::Point2f(-forward.y,forward.x);
  }
};

std::optional<ArmAxes> armAxes(const Keypoints& detection) {
  const auto center=detection.armorCenter();
  if (!detection.point_valid[4] || !center) return {};
  const auto delta=*center-detection.center_r;
  const auto length=float(cv::norm(delta));
  if (!std::isfinite(length) || length<12) return {};
  return ArmAxes{detection.center_r,delta*(1.F/length),length};
}

bool inside(cv::Point2f point,cv::Size size) {
  return point.x>=0 && point.y>=0 && point.x<size.width-1 && point.y<size.height-1;
}

std::optional<Corners> numberedCorners(const DetectedContours& features) {
  if (features.fan_points.size()!=4) return {};
  Corners points;
  std::array<bool,4> seen{};
  for (std::size_t i=0;i<points.size();++i) {
    const int id=features.fanPointId(i);
    if (!features.fanPointValid(i) || id<0 || id>=int(points.size()) || seen[id]) return {};
    points[id]=features.fan_points[i];
    seen[id]=true;
  }
  return points;
}

struct Edge {
  float slope;
  float offset;

  float at(float along) const { return offset+slope*along; }
};

struct ArmShape {
  std::array<Edge,2> sides;
  float inner;
  float outer;
};

struct FittedSide {
  Edge line;
  float first;
  float last;
};

std::optional<ArmShape> armShape(const Corners& points,const ArmAxes& axes) {
  Corners local;
  for (std::size_t i=0;i<points.size();++i) local[i]=axes.local(points[i]);
  const float inner=(local[2].x+local[3].x)*.5F,outer=(local[0].x+local[1].x)*.5F;
  if (!std::isfinite(inner) || !std::isfinite(outer) || outer-inner<12) return {};
  const auto side=[&](int outer_id,int inner_id) -> std::optional<Edge> {
    const auto a=local[outer_id],b=local[inner_id];
    const auto delta=a-b;
    if (delta.x<8) return {};
    const float slope=delta.y/delta.x;
    if (!std::isfinite(slope) || std::abs(slope)>.65F) return {};
    return Edge{slope,a.y-slope*a.x};
  };
  // 编号 0、1 为外端，2、3 为内端。左右的正负由已有编号约定决定。
  const auto left=side(0,3),right=side(1,2);
  if (!left || !right) return {};
  return ArmShape{{*left,*right},inner,outer};
}

float colorDifference(const cv::Mat& bgr,cv::Point2f point,int main) {
  const int x=int(point.x),y=int(point.y);
  const float fx=point.x-x,fy=point.y-y;
  const auto* top=bgr.ptr<cv::Vec3b>(y)+x;
  const auto* bottom=bgr.ptr<cv::Vec3b>(y+1)+x;
  const auto contrast=[main](const cv::Vec3b& value) { return int(value[main])-value[2-main]; };
  return (1-fy)*((1-fx)*contrast(top[0])+fx*contrast(top[1]))+
         fy*((1-fx)*contrast(bottom[0])+fx*contrast(bottom[1]));
}

std::optional<FittedSide> fitSide(const cv::Mat& bgr,int channel,const ArmAxes& axes,
                                  const ArmShape& shape,int side,float width,float outward) {
  constexpr int rows=28;
  const float span=shape.outer-shape.inner;
  const float radius=std::clamp(.32F*width,3.F,14.F);
  const int extent=int(std::ceil(radius));
  std::vector<cv::Point2f> samples;
  samples.reserve(rows);
  for (int i=0;i<rows;++i) {
    const float along=shape.inner+span*(.08F+.84F*i/(rows-1));
    const float expected=shape.sides[side].at(along);
    std::array<float,31> values{};
    if (!inside(axes.pixel(along,expected-extent-1),bgr.size()) ||
        !inside(axes.pixel(along,expected+extent+1),bgr.size())) continue;
    for (int j=-extent-1;j<=extent+1;++j)
      values[j+extent+1]=colorDifference(bgr,axes.pixel(along,expected+outward*j),channel);
    const auto gradient=[&](int j) { return values[j+extent]-values[j+extent+2]; };
    float strongest=0,best_position=0;
    for (int j=-extent+1;j<extent;++j) {
      const float g=gradient(j),before=gradient(j-1),after=gradient(j+1);
      if (g<17 || g<before || g<after || values[j+extent]<16) continue;
      const double across=expected+outward*j;
      const auto a=axes.pixel(along-1,float(across)),b=axes.pixel(along+1,float(across));
      if (!inside(a,bgr.size()) || !inside(b,bgr.size()) ||
          std::abs(colorDifference(bgr,a,channel)-colorDifference(bgr,b,channel))>.65F*g) continue;
      const float score=g/(1+.035F*std::abs(j));
      if (score<=strongest) continue;
      const float curvature=before-2*g+after;
      const float offset=curvature<-.01F ? std::clamp(.5F*(before-after)/curvature,-.5F,.5F) : 0;
      best_position=expected+outward*(j+offset);
      strongest=score;
    }
    if (strongest>0) samples.emplace_back(along,best_position);
  }
  if (samples.size()<6) return {};

  cv::Vec4f fitted;
  cv::fitLine(samples,fitted,cv::DIST_HUBER,0,.02,.01);
  if (std::abs(fitted[0])<.85F) return {};
  const float slope=fitted[1]/fitted[0];
  if (!std::isfinite(slope) || std::abs(slope)>.48F) return {};
  const Edge edge{slope,fitted[3]-slope*fitted[2]};
  const float tolerance=std::clamp(.06F*width,1.35F,3.F);
  float first=shape.outer,last=shape.inner;
  int inliers=0;
  for (const auto& p:samples) {
    if (std::abs(p.y-edge.at(p.x))>tolerance) continue;
    first=std::min(first,p.x);
    last=std::max(last,p.x);
    ++inliers;
  }
  if (inliers<6 || inliers*3<int(samples.size())*2 || last-first<.42F*span ||
      std::abs(edge.at(shape.inner)-shape.sides[side].at(shape.inner))>radius ||
      std::abs(edge.at(shape.outer)-shape.sides[side].at(shape.outer))>radius)
    return {};
  return FittedSide{edge,first,last};
}

bool validShape(const ArmShape& shape,float width,float orientation) {
  if (!std::isfinite(shape.inner) || !std::isfinite(shape.outer) || shape.outer-shape.inner<12)
    return false;
  for (float along:{shape.inner,shape.outer}) {
    const float span=orientation*(shape.sides[1].at(along)-shape.sides[0].at(along));
    if (!std::isfinite(span) || span<.42F*width || span>1.65F*width) return false;
  }
  return true;
}

Corners corners(const ArmShape& shape,const ArmAxes& axes) {
  return {axes.pixel(shape.outer,shape.sides[0].at(shape.outer)),
          axes.pixel(shape.outer,shape.sides[1].at(shape.outer)),
          axes.pixel(shape.inner,shape.sides[1].at(shape.inner)),
          axes.pixel(shape.inner,shape.sides[0].at(shape.inner))};
}
} // namespace

void refineInactiveArm(const cv::Mat& bgr,Color color,const Keypoints& current,
                       const Keypoints& raw,const Keypoints* previous,TimePoint now,
                       double max_gap_s,float measurement_weight,DetectedContours& output) {
  output.inactive_arm_outline.reset();
  if (current.class_id!=0 || !raw.image_contours ||
      current.target_ring_status==TargetRingStatus::Rejected) return;
  const auto axes=armAxes(current);
  const auto raw_points=numberedCorners(*raw.image_contours);
  if (!axes || !raw_points) return;
  const auto rough=armShape(*raw_points,*axes);
  if (!rough) return;
  const float middle=(rough->inner+rough->outer)*.5F;
  const float signed_width=rough->sides[1].at(middle)-rough->sides[0].at(middle);
  const float width=std::abs(signed_width);
  const float orientation=std::copysign(1.F,signed_width);
  if (width<5 || !validShape(*rough,width,orientation)) return;

  auto observed=*rough;
  std::array<std::optional<FittedSide>,2> fits;
  const int channel=color==Color::Blue ? 0 : 2;
  for (int i=0;i<2;++i) {
    fits[i]=fitSide(bgr,channel,*axes,*rough,i,width,i==0 ? -orientation : orientation);
    if (fits[i]) observed.sides[i]=fits[i]->line;
  }
  if (!validShape(observed,width,orientation)) {
    observed=*rough;
    fits={};
  }
  const auto stable_points=numberedCorners(output);
  const auto stable_shape=stable_points ? armShape(*stable_points,*axes) : std::nullopt;
  ArmShape display=observed;
  if (stable_shape && validShape(*stable_shape,width,orientation)) {
    display.inner=stable_shape->inner;
    display.outer=stable_shape->outer;
    for (int i=0;i<2;++i)
      if (!fits[i]) display.sides[i]=stable_shape->sides[i];
  }

  InactiveArmOutline outline{};
  for (int i=0;i<2;++i) {
    if (const auto& fit=fits[i]) {
      outline.measured_edges[i]=std::array{axes->pixel(fit->first,fit->line.at(fit->first)),
                                           axes->pixel(fit->last,fit->line.at(fit->last))};
      outline.side_observed_at[i]=now;
    }
  }
  if (previous && previous->image_contours && previous->image_contours->inactive_arm_outline) {
    const auto old_axes=armAxes(*previous);
    const auto& old_outline=*previous->image_contours->inactive_arm_outline;
    const auto old=old_axes ? armShape(old_outline.corners,*old_axes) : std::nullopt;
    if (old && axes->forward.dot(old_axes->forward)>.94F) {
      const float scale=axes->length/old_axes->length;
      if (scale>.8F && scale<1.25F && std::abs(old->inner*scale-display.inner)<.35F*(rough->outer-rough->inner) &&
          std::abs(old->outer*scale-display.outer)<.35F*(rough->outer-rough->inner)) {
        const float weight=std::clamp(measurement_weight,.12F,.55F);
        display.inner=std::lerp(old->inner*scale,display.inner,weight);
        display.outer=std::lerp(old->outer*scale,display.outer,weight);
        for (int i=0;i<2;++i) {
          const auto seen=old_outline.side_observed_at[i];
          const double age=std::chrono::duration<double>(now-seen).count();
          const Edge carried{old->sides[i].slope,old->sides[i].offset*scale};
          const bool recently_observed=seen!=TimePoint{} && age>=0 && age<=max_gap_s;
          if (std::abs(carried.at(middle)-display.sides[i].at(middle))>.48F*width ||
              (!recently_observed && (!stable_shape || !validShape(*stable_shape,width,orientation))))
            continue;
          // 当稳定轮廓仍在时，即使一侧暂时缺少直接边缘，也可以用本帧轮廓作为低置信输入。
          display.sides[i]={std::lerp(carried.slope,display.sides[i].slope,weight),
                            std::lerp(carried.offset,display.sides[i].offset,weight)};
          if (!fits[i] && recently_observed) outline.side_observed_at[i]=seen;
        }
      }
    }
  }
  if (!validShape(display,width,orientation)) return;
  outline.corners=corners(display,*axes);
  if (std::all_of(outline.corners.begin(),outline.corners.end(),
                  [&](cv::Point2f point) { return inside(point,bgr.size()); }))
    output.inactive_arm_outline=outline;
}
} // namespace Rune::detail
