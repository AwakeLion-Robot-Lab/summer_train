#include "rune/f1-observation/image_refiner.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Rune {
namespace {
std::array<cv::Point2f,4> targetPoints(const Keypoints& k) {
  return {k.top,k.left,k.right,k.bottom};
}
bool hasCoarseRing(const Keypoints& k) {
  const auto& p=k.target_ring_coarse_points;
  return k.target_ring_status==TargetRingStatus::Verified &&
      cv::norm(p[0]-p[3])>1 && cv::norm(p[1]-p[2])>1;
}
std::array<cv::Point2f,4> registrationPoints(const Keypoints& k) {
  return hasCoarseRing(k) ? k.target_ring_coarse_points : targetPoints(k);
}
cv::Point2f targetCenter(const Keypoints& k) {
  const auto p=registrationPoints(k);
  return (p[0]+p[1]+p[2]+p[3])*.25F;
}
bool fullTarget(const Keypoints& k) {
  return std::all_of(k.point_valid.begin(),k.point_valid.begin()+4,[](bool v){return v;});
}
bool finite(const cv::Point2f& p) { return std::isfinite(p.x) && std::isfinite(p.y); }
double diameter(const Keypoints& k) {
  const auto points=registrationPoints(k);
  double result=0;
  for (int i=0;i<4;++i) for (int j=0;j<i;++j)
    if (k.point_valid[i] && k.point_valid[j]) result=std::max(result,cv::norm(points[i]-points[j]));
  return result;
}
std::optional<cv::Matx23f> opticalFlow(const cv::Mat& before,const cv::Mat& after,
    const cv::Rect& roi,bool target) {
  std::vector<cv::Point2f> points;
  cv::goodFeaturesToTrack(before(roi),points,40,.02,4);
  if (points.size()<6) return {};
  for (auto& p : points) p+=cv::Point2f(roi.tl());
  std::vector<cv::Point2f> forward,backward;
  std::vector<uchar> forward_valid,backward_valid;
  std::vector<float> errors,backward_errors;
  const cv::TermCriteria criteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,.01);
  cv::calcOpticalFlowPyrLK(before,after,points,forward,forward_valid,errors,{21,21},3,criteria);
  cv::calcOpticalFlowPyrLK(after,before,forward,backward,backward_valid,backward_errors,{21,21},3,criteria);
  std::vector<cv::Point2f> source,destination;
  for (std::size_t i=0;i<points.size();++i) {
    if (!forward_valid[i] || !backward_valid[i] || errors[i]>15 ||
        !finite(forward[i]) || !finite(backward[i]) || cv::norm(backward[i]-points[i])>.5) continue;
    source.push_back(points[i]); destination.push_back(forward[i]);
  }
  if (source.size()<6 || source.size()*2<points.size()) return {};
  cv::Mat inliers;
  const auto affine=cv::estimateAffinePartial2D(source,destination,inliers,cv::RANSAC,.75);
  if (affine.empty() || !cv::checkRange(affine) || cv::countNonZero(inliers)<6 ||
      cv::countNonZero(inliers)<.8*source.size()) return {};
  cv::Matx23f transform;
  for (int i=0;i<2;++i) for (int j=0;j<3;++j) transform(i,j)=float(affine.at<double>(i,j));
  const float scale=std::hypot(transform(0,0),transform(1,0));
  if (scale<.9F || scale>1.1F ||
      (!target && (std::abs(scale-1)>.01F || std::abs(transform(1,0))>.01F))) return {};
  return transform;
}
std::optional<cv::Matx23f> registration(const cv::Mat& before,const cv::Mat& after,
    cv::Point2f center,int half_size,bool target,double min_correlation,
    cv::Point2f displacement,const std::optional<cv::Matx23f>& initial_motion={},
    int iterations=40) {
  if (!finite(center) || center.x<0 || center.y<0 || center.x>=before.cols || center.y>=before.rows)
    return {};
  const int x=cvRound(center.x),y=cvRound(center.y);
  const int x0=std::max(0,x-half_size),y0=std::max(0,y-half_size);
  const int x1=std::min(before.cols,x+half_size),y1=std::min(before.rows,y+half_size);
  if (x1-x0<20 || y1-y0<20) return {};
  const cv::Rect roi(x0,y0,x1-x0,y1-y0);
  // First retain the ordinary identity initialization. If it fails, use the
  // observed displacement to enter ECC's convergence basin. The NN offset is
  // only an initial guess: the image correlation and geometry gates still
  // have to pass before any coordinate can be transferred.
  for (int attempt=0;attempt<(initial_motion ? 1 : 2);++attempt) {
    if (attempt==1 && (!finite(displacement) || cv::norm(displacement)<2)) break;
    cv::Mat warp=cv::Mat::eye(2,3,CV_32F);
    if (initial_motion) {
      for (int i=0;i<2;++i) for (int j=0;j<3;++j) warp.at<float>(i,j)=(*initial_motion)(i,j);
      // Full-image motion -> the local coordinates used by ECC.
      warp.at<float>(0,2)+=(*initial_motion)(0,0)*x0+(*initial_motion)(0,1)*y0-x0;
      warp.at<float>(1,2)+=(*initial_motion)(1,0)*x0+(*initial_motion)(1,1)*y0-y0;
    } else if (attempt==1) {
      warp.at<float>(0,2)=displacement.x;
      warp.at<float>(1,2)=displacement.y;
    }
    try {
      const double correlation=cv::findTransformECC(before(roi),after(roi),warp,
          target ? cv::MOTION_AFFINE : cv::MOTION_TRANSLATION,
          {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,iterations,1e-5});
      if (!std::isfinite(correlation) || correlation<min_correlation || !cv::checkRange(warp)) continue;
      cv::Matx23f transform;
      for (int i=0;i<2;++i) for (int j=0;j<3;++j) transform(i,j)=warp.at<float>(i,j);
      const cv::Matx22f linear(transform(0,0),transform(0,1),transform(1,0),transform(1,1));
      if (target) {
        cv::Mat singular;
        cv::SVD::compute(cv::Mat(linear),singular);
        if (singular.at<float>(0)>1.1F || singular.at<float>(1)<.9F || cv::determinant(linear)<=0) continue;
      }
      const cv::Vec2f origin{float(x0),float(y0)};
      const cv::Vec2f translation=cv::Vec2f(transform(0,2),transform(1,2))+origin-linear*origin;
      transform(0,2)=translation[0]; transform(1,2)=translation[1];
      return transform;
    } catch (const cv::Exception&) { /* Retry with the measured displacement. */ }
  }
  // The caller already has image-verified body motion; do not spend another
  // optical-flow pass chasing an animated or newly occluded lamp texture.
  if (initial_motion) return {};
  // LEDs can change brightness enough to fail ECC even with valid image
  // motion. Independently verified corner tracks bridge those frames; require
  // forward/backward consistency, photometric quality and a common transform.
  try { return opticalFlow(before,after,roi,target); }
  catch (const cv::Exception&) { return {}; }
}
cv::Point2f transfer(const cv::Matx23f& transform,const cv::Point2f& p) {
  const auto v=transform*cv::Vec3f(p.x,p.y,1);
  return {v[0],v[1]};
}
bool inside(const cv::Point2f& p,const cv::Size& size) {
  return finite(p) && p.x>=0 && p.y>=0 && p.x<size.width && p.y<size.height;
}

cv::Matx22f ellipseShape(const cv::RotatedRect& e) {
  const float a=e.angle*float(CV_PI/180),c=std::cos(a),s=std::sin(a);
  const cv::Matx22f rotation(c,-s,s,c);
  return rotation*cv::Matx22f(e.size.width*e.size.width*.25F,0,
                            0,e.size.height*e.size.height*.25F)*rotation.t();
}
cv::RotatedRect ellipseFromShape(cv::Point2f center,const cv::Matx22f& shape) {
  cv::Mat values,vectors;
  cv::eigen(cv::Mat(shape),values,vectors);
  return {center,{2*std::sqrt(std::max(0.F,values.at<float>(0))),
                  2*std::sqrt(std::max(0.F,values.at<float>(1)))},
          float(std::atan2(vectors.at<float>(0,1),vectors.at<float>(0,0))*180/CV_PI)};
}

void stabilizeTargetFeatures(DetectedContours& out,const DetectedContours& previous,
    const cv::Matx23f& warp,float alpha,const cv::Size& image_size,
    ImageRefinerDiagnostics& diagnostics) {
  if (!out.target_ellipse || !previous.target_ellipse) return;
  const auto current=*out.target_ellipse;
  const auto center=transfer(warp,previous.target_ellipse->center);
  const cv::Matx22f linear(warp(0,0),warp(0,1),warp(1,0),warp(1,1));
  const auto shape=linear*ellipseShape(*previous.target_ellipse)*linear.t();
  const auto projected=ellipseFromShape(center,shape);
  const double diameter=std::max(current.size.width,current.size.height);
  const double gate=std::max(2.0,.12*diameter);
  // Compare invariant axes, not ellipse.angle: a near-circle can change its
  // reported axis angle by 90 degrees without any geometric motion.
  const double current_major=std::max(current.size.width,current.size.height);
  const double current_minor=std::min(current.size.width,current.size.height);
  if (!inside(center,image_size) || cv::norm(center-current.center)>gate ||
      std::abs(current_major-projected.size.width)>std::max(2.0,.12*current_major) ||
      std::abs(current_minor-projected.size.height)>std::max(2.0,.12*current_minor)) {
    out.target_ellipse.reset(); out.gap_points={};
    ++diagnostics.rejected_feature_points;
    return;
  }
  out.target_ellipse=ellipseFromShape(alpha*current.center+(1-alpha)*center,
      alpha*ellipseShape(current)+(1-alpha)*shape);
  ++diagnostics.refined_feature_targets;
  for (std::size_t i=0;i<out.gap_points.size();++i) {
    if (!out.gap_points[i] || !previous.gap_points[i]) continue;
    const auto predicted=transfer(warp,*previous.gap_points[i]);
    if (!inside(predicted,image_size) || cv::norm(predicted-*out.gap_points[i])>gate) {
      out.gap_points[i].reset(); ++diagnostics.rejected_feature_points;
    } else out.gap_points[i]=alpha*(*out.gap_points[i])+(1-alpha)*predicted;
  }
}

void stabilizeFanFeatures(DetectedContours& out,const DetectedContours& previous,
    const DetectedContours& raw_previous,const cv::Mat& before,const cv::Mat& after,
    float alpha,double min_correlation,double target_diameter,
    const std::optional<cv::Matx23f>& body_motion,
    ImageRefinerDiagnostics& diagnostics) {
  if (out.fan_points.empty() || previous.fan_points.empty() || raw_previous.fan_points.empty()) return;
  struct Pair { std::size_t now,old; cv::Point2f shift; };
  std::vector<Pair> pairs;
  cv::Point2f center{};
  std::vector<float> dx,dy;
  for (std::size_t i=0;i<out.fan_points.size();++i) {
    if (!out.fanPointValid(i) || !finite(out.fan_points[i])) continue;
    for (std::size_t j=0;j<raw_previous.fan_points.size();++j) {
      if (raw_previous.fanPointId(j)!=out.fanPointId(i) || !finite(raw_previous.fan_points[j])) continue;
      const auto shift=out.fan_points[i]-raw_previous.fan_points[j];
      pairs.push_back({i,j,shift}); center+=raw_previous.fan_points[j];
      dx.push_back(shift.x);dy.push_back(shift.y);break;
    }
  }
  if (pairs.size()<3) return;
  center*=1.F/pairs.size();
  const auto median=[](std::vector<float>& values) {
    const auto middle=values.begin()+values.size()/2;
    std::nth_element(values.begin(),middle,values.end());return *middle;
  };
  const cv::Point2f displacement(median(dx),median(dy));
  double radius=0;
  for (const auto& pair : pairs)
    radius=std::max(radius,cv::norm(raw_previous.fan_points[pair.old]-center));
  const int half=int(std::clamp(std::ceil(radius+15),20.0,256.0));
  // The fan gets its own local registration. A rotating LED pattern inside a
  // circular target must not rotate the remote fan endpoints around that target.
  auto warp=registration(before,after,center,half,true,min_correlation,displacement,body_motion,20);
  // Animated lamp LEDs can defeat local image registration. A target-image-
  // verified similarity from the measured R/target centers preserves blade
  // motion without importing the circular target texture's apparent rotation.
  if (!warp) warp=body_motion;
  if (!warp) return;
  out.fan_point_valid.assign(out.fan_points.size(),true);
  const double gate=std::max(2.0,.12*target_diameter);
  for (const auto& pair : pairs) {
    for (std::size_t j=0;j<previous.fan_points.size();++j) {
      if (previous.fanPointId(j)!=out.fanPointId(pair.now) || !previous.fanPointValid(j)) continue;
      const auto predicted=transfer(*warp,previous.fan_points[j]);
      if (!inside(predicted,after.size()) || cv::norm(predicted-out.fan_points[pair.now])>gate) {
        out.fan_point_valid[pair.now]=false;++diagnostics.rejected_feature_points;
      } else {
        out.fan_points[pair.now]=alpha*out.fan_points[pair.now]+(1-alpha)*predicted;
        ++diagnostics.refined_fan_points;
      }
      break;
    }
  }
}

std::optional<cv::Matx23f> bladeImageMotion(const Keypoints& before,const Keypoints& after,
    const std::optional<cv::Matx23f>& target_warp) {
  if (!target_warp || !fullTarget(before) || !fullTarget(after) ||
      !before.point_valid[4] || !after.point_valid[4]) return {};
  const auto old_center=targetCenter(before),new_center=targetCenter(after);
  const auto a=old_center-before.center_r,b=new_center-after.center_r;
  const double length=a.dot(a);
  if (!finite(a) || !finite(b) || length<25 ||
      cv::norm(transfer(*target_warp,old_center)-new_center)>std::max(2.0,.05*diameter(after))) return {};
  const float c=float(a.dot(b)/length),s=float(a.cross(b)/length);
  const float scale=std::hypot(c,s);
  if (scale<.9F || scale>1.1F || std::abs(std::atan2(s,c))>.25) return {};
  // Translation comes from verified image motion, not the new noisy centroid.
  // Only orientation/scale comes from the R-to-target geometry.
  const auto p=old_center,q=transfer(*target_warp,old_center);
  return cv::Matx23f(c,-s,q.x-c*p.x+s*p.y,s,c,q.y-s*p.x-c*p.y);
}
} // namespace

ImageRefinerConfig ImageRefinerConfig::load(const std::filesystem::path& yaml) {
  const auto root=YAML::LoadFile(yaml.string());
  const auto rune=root["rune"];
  return fromYaml(rune ? rune["image_refiner"] : YAML::Node{});
}
ImageRefinerConfig ImageRefinerConfig::fromYaml(const YAML::Node& node) {
  ImageRefinerConfig config;
  if (node && !node.IsNull()) {
    if (!node.IsMap()) throw std::invalid_argument("rune.image_refiner must be a map");
    config.enabled=node["enabled"].as<bool>(config.enabled);
    config.max_gap_s=node["max_gap_s"].as<double>(config.max_gap_s);
    config.measurement_time_constant_s=node["measurement_time_constant_s"].as<double>(config.measurement_time_constant_s);
    config.min_correlation=node["min_correlation"].as<double>(config.min_correlation);
  }
  (void)ImageRefiner(config); // Validate once for runtime and standalone callers.
  return config;
}
ImageRefiner::ImageRefiner(ImageRefinerConfig config) : config_(config) {
  if (!std::isfinite(config.max_gap_s) || config.max_gap_s<=0 || config.max_gap_s>1 ||
      !std::isfinite(config.measurement_time_constant_s) || config.measurement_time_constant_s<=0 ||
      !std::isfinite(config.min_correlation) || config.min_correlation<=0 || config.min_correlation>1)
    throw std::invalid_argument("invalid rune image refiner configuration");
}
void ImageRefiner::reset() {
  previous_gray_.release(); previous_detections_.clear(); previous_raw_detections_.clear();
  previous_ring_precision_times_.clear();
  previous_coarse_filtered_.clear();previous_ring_center_corrections_.clear();
  previous_time_={}; diagnostics_={};
}
void ImageRefiner::update(Frame& frame) {
  diagnostics_={};
  if (!config_.enabled) { reset(); return; }
  if (frame.bgr.empty() || frame.bgr.type()!=CV_8UC3 || frame.captured_at==TimePoint{}) { reset(); return; }
  const double dt=std::chrono::duration<double>(frame.captured_at-previous_time_).count();
  if (dt<=0 || dt>config_.max_gap_s || frame.bgr.size()!=previous_gray_.size() ||
      frame.mode!=previous_mode_ || frame.color!=previous_color_) reset();
  // No detection means no observed measurement: retain only bounded history.
  if (frame.detections.empty()) return;
  auto raw_detections=frame.detections;
  std::vector<std::array<cv::Point2f,4>> coarse_filtered(frame.detections.size());
  std::vector<cv::Point2f> ring_center_corrections(frame.detections.size());
  std::vector<TimePoint> ring_precision_times(frame.detections.size());
  for (std::size_t i=0;i<frame.detections.size();++i) {
    const auto& k=frame.detections[i];
    if (hasCoarseRing(k)) {
      coarse_filtered[i]=k.target_ring_coarse_points;
      if (fullTarget(k)) ring_center_corrections[i]=*k.armorCenter()-targetCenter(k);
    }
    if (!k.target_ring_subpixel || !hasCoarseRing(k)) continue;
    ring_precision_times[i]=frame.captured_at;
  }
  cv::Mat gray;
  cv::cvtColor(frame.bgr,gray,cv::COLOR_BGR2GRAY);
  if (!previous_gray_.empty()) {
    const float alpha=static_cast<float>(-std::expm1(-dt/config_.measurement_time_constant_s));
    std::vector<int> feature_associations(frame.detections.size(),-1);
    std::vector<std::optional<cv::Matx23f>> feature_warps(frame.detections.size());
    for (std::size_t index=0;index<frame.detections.size();++index) {
      auto& current=frame.detections[index];
      const double size=diameter(current),gate=std::max(2.0,.2*size);
      // R is shared across blades. A target switch must not move the hub to a
      // different NN candidate's regressed R position.
      const Keypoints* hub=nullptr;
      const Keypoints* raw_hub=nullptr;
      double nearest=std::max(30.0,size);
      if (current.point_valid[4] && finite(current.center_r)) {
        for (std::size_t i=0;i<previous_raw_detections_.size();++i) {
          const auto& old=previous_raw_detections_[i];
          if (!old.point_valid[4]) continue;
          const double distance=cv::norm(current.center_r-old.center_r);
          if (distance<nearest) {
            nearest=distance; hub=&previous_detections_[i]; raw_hub=&old;
          }
        }
        if (hub) {
          const auto warp=registration(previous_gray_,gray,(raw_hub->center_r+current.center_r)*.5F,
                                       40,false,config_.min_correlation,
                                       current.center_r-raw_hub->center_r);
          if (warp) {
            const auto p=transfer(*warp,hub->center_r);
            if (inside(p,gray.size()) && cv::norm(p-current.center_r)<gate) {
              current.center_r=alpha*current.center_r+(1-alpha)*p;
              ++diagnostics_.refined_r;
            }
          }
        }
      }
      if (!fullTarget(current)) continue;
      const auto center=targetCenter(current);
      if (!finite(center)) continue;
      const Keypoints* target=nullptr;
      const Keypoints* raw_target=nullptr;
      nearest=std::max(30.0,.5*size);
      for (std::size_t i=0;i<previous_raw_detections_.size();++i) {
        const auto& old=previous_raw_detections_[i];
        if (old.class_id!=current.class_id || !fullTarget(old)) continue;
        const auto old_center=targetCenter(old);
        const double distance=cv::norm(center-old_center);
        if (distance<nearest) {
          nearest=distance; target=&previous_detections_[i]; raw_target=&old;
        }
      }
      if (!target) continue;
      feature_associations[index]=int(target-previous_detections_.data());
      const auto before=targetPoints(*target);
      auto now=targetPoints(current);
      const auto old_center=targetCenter(*raw_target);
      double radius=0;
      for (const auto& p : registrationPoints(*raw_target)) radius=std::max(radius,cv::norm(p-old_center));
      const int half=static_cast<int>(std::clamp(std::ceil(radius+15),20.0,128.0));
      const auto warp=registration(previous_gray_,gray,old_center,half,true,config_.min_correlation,
                                   center-old_center);
      feature_warps[index]=warp;
      if (!warp) continue;
      const auto old_index=std::size_t(target-previous_detections_.data());
      bool precise_update=false;
      if (hasCoarseRing(current) && hasCoarseRing(*raw_target)) {
        for (int j=0;j<4;++j)
          coarse_filtered[index][j]=alpha*current.target_ring_coarse_points[j]+
              (1-alpha)*transfer(*warp,previous_coarse_filtered_[old_index][j]);
        const auto last_precise=previous_ring_precision_times_[old_index];
        const double age=std::chrono::duration<double>(frame.captured_at-last_precise).count();
        const bool measured=current.target_ring_subpixel;
        const bool recent=last_precise!=TimePoint{} && age>=0 && age<=config_.max_gap_s;
        const auto& base=coarse_filtered[index];
        if ((measured || recent) && cv::norm(base[0]-base[3])>2 && cv::norm(base[1]-base[2])>2) {
          // Keep the semantic directions and center anchored to the ordinary
          // geometry tracker. Animated LEDs can make local registration rotate
          // a stationary target; it must not rotate the semantic points away
          // from their independently observed radial/tangential directions.
          const float beta=measured ? float(-std::expm1(-dt/.5)) : 0;
          const auto base_center=(base[0]+base[1]+base[2]+base[3])*.25F;
          const auto old_bias=previous_ring_center_corrections_[old_index];
          const cv::Point2f predicted_bias((*warp)(0,0)*old_bias.x+(*warp)(0,1)*old_bias.y,
                                          (*warp)(1,0)*old_bias.x+(*warp)(1,1)*old_bias.y);
          const auto measured_bias=(now[0]+now[1]+now[2]+now[3])*.25F-targetCenter(current);
          auto bias=beta*measured_bias+(1-beta)*predicted_bias;
          const double max_bias=.05*std::min(cv::norm(base[0]-base[3]),cv::norm(base[1]-base[2]));
          if (cv::norm(bias)>max_bias) bias*=float(max_bias/cv::norm(bias));
          const auto precise_center=base_center+bias;
          for (const auto [a,b] : {std::pair{0,3},std::pair{1,2}}) {
            auto direction=base[a]-base[b];
            const float base_radius=float(cv::norm(direction)*.5);
            direction*=1/(2*base_radius);
            const auto old=(before[a]-before[b])*.5F;
            const cv::Point2f projected((*warp)(0,0)*old.x+(*warp)(0,1)*old.y,
                                        (*warp)(1,0)*old.x+(*warp)(1,1)*old.y);
            const float radius=std::clamp(beta*float(cv::norm(now[a]-now[b])*.5)+
                (1-beta)*float(cv::norm(projected)),.88F*base_radius,1.04F*base_radius);
            now[a]=precise_center+direction*radius;
            now[b]=precise_center-direction*radius;
          }
          if (!measured) ring_precision_times[index]=last_precise;
          precise_update=true;
        }
      }
      std::array<cv::Point2f,4> predicted;
      bool valid=true;
      for (int j=0;j<4;++j) {
        predicted[j]=transfer(*warp,before[j]);
        valid &= inside(predicted[j],gray.size()) &&
            cv::norm(predicted[j]-targetPoints(current)[j])<gate;
      }
      if (current.target_ring_status==TargetRingStatus::Verified &&
          target->target_ring_status==TargetRingStatus::Verified) {
        // Compare diameters AFTER independently measuring image motion. Real
        // approach, rotation and foreshortening move both predictions together;
        // switching to a nested ring changes the measured diameters alone.
        bool consistent=valid;
        auto reference=predicted,measurement=now;
        if (hasCoarseRing(current) && hasCoarseRing(*raw_target)) {
          reference=raw_target->target_ring_coarse_points;
          for (auto& p : reference) p=transfer(*warp,p);
          measurement=current.target_ring_coarse_points;
        }
        for (const auto [a,b] : {std::pair{0,3},std::pair{1,2}}) {
          const double length=cv::norm(reference[a]-reference[b]);
          consistent &= std::abs(cv::norm(measurement[a]-measurement[b])-length)<=std::max(2.0,.08*length);
        }
        if (!consistent) {
          std::fill_n(current.point_valid.begin(),4,false);
          current.target_ring_status=TargetRingStatus::Rejected;
          std::fill_n(raw_detections[index].point_valid.begin(),4,false);
          raw_detections[index].target_ring_status=TargetRingStatus::Rejected;
          ++diagnostics_.rejected_target_rings;
          continue;
        }
      }
      if (!valid) continue;
      std::array<cv::Point2f*,4> output{&current.top,&current.left,&current.right,&current.bottom};
      for (int j=0;j<4;++j)
        *output[j]=precise_update ? now[j] : alpha*now[j]+(1-alpha)*predicted[j];
      if (hasCoarseRing(current)) {
        const auto& base=coarse_filtered[index];
        ring_center_corrections[index]=*current.armorCenter()-(base[0]+base[1]+base[2]+base[3])*.25F;
      }
      ++diagnostics_.refined_targets;
    }
    for (std::size_t i=0;i<frame.detections.size();++i) {
      auto& current=frame.detections[i];
      if (!current.image_contours) continue;
      // Shared raw contour data is immutable. Never filter in place: the raw
      // association history and callers may still own this exact measurement.
      auto features=std::make_shared<DetectedContours>(*current.image_contours);
      if (current.target_ring_status==TargetRingStatus::Rejected) {
        features->target_ellipse.reset();features->gap_points={};
      }
      const int old_index=feature_associations[i];
      if (old_index>=0) {
        const auto& old=previous_detections_[old_index].image_contours;
        const auto& raw_old=previous_raw_detections_[old_index].image_contours;
        if (old && raw_old) {
          if (feature_warps[i]) stabilizeTargetFeatures(*features,*old,*feature_warps[i],alpha,gray.size(),diagnostics_);
          stabilizeFanFeatures(*features,*old,*raw_old,previous_gray_,gray,alpha,
                               config_.min_correlation,diameter(raw_detections[i]),
                               bladeImageMotion(previous_raw_detections_[old_index],raw_detections[i],
                                                feature_warps[i]),diagnostics_);
        }
      }
      current.image_contours=std::move(features);
    }
  }
  const bool rejected=std::any_of(frame.detections.begin(),frame.detections.end(),[](const auto& k) {
    return k.target_ring_status==TargetRingStatus::Rejected;
  });
  const bool observed_target=std::any_of(frame.detections.begin(),frame.detections.end(),[&](const auto& k) {
    return fullTarget(k) && !(frame.mode==Mode::Big && k.class_id==1);
  });
  // A failed ring check must not become the next frame's reference. Retain the
  // last good image only until max_gap_s; publish no historical target points.
  if (rejected && !observed_target) return;
  previous_gray_=std::move(gray); previous_detections_=frame.detections;
  previous_raw_detections_=std::move(raw_detections);
  previous_ring_precision_times_=std::move(ring_precision_times);
  previous_coarse_filtered_=std::move(coarse_filtered);
  previous_ring_center_corrections_=std::move(ring_center_corrections);
  previous_time_=frame.captured_at; previous_mode_=frame.mode; previous_color_=frame.color;
}
} // namespace Rune
