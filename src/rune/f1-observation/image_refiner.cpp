#include "rune/f1-observation/image_refiner.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace Rune {
namespace {
std::array<cv::Point2f,4> targetPoints(const Keypoints& k) {
  return {k.top,k.left,k.right,k.bottom};
}
bool fullTarget(const Keypoints& k) {
  return std::all_of(k.point_valid.begin(),k.point_valid.begin()+4,[](bool v){return v;});
}
bool finite(const cv::Point2f& p) { return std::isfinite(p.x) && std::isfinite(p.y); }
double diameter(const Keypoints& k) {
  const auto points=targetPoints(k);
  double result=0;
  for (int i=0;i<4;++i) for (int j=0;j<i;++j)
    if (k.point_valid[i] && k.point_valid[j]) result=std::max(result,cv::norm(points[i]-points[j]));
  return result;
}
std::optional<cv::Matx23f> registration(const cv::Mat& before,const cv::Mat& after,
    cv::Point2f center,int half_size,bool target,double min_correlation) {
  if (!finite(center) || center.x<0 || center.y<0 || center.x>=before.cols || center.y>=before.rows)
    return {};
  const int x=cvRound(center.x),y=cvRound(center.y);
  const int x0=std::max(0,x-half_size),y0=std::max(0,y-half_size);
  const int x1=std::min(before.cols,x+half_size),y1=std::min(before.rows,y+half_size);
  if (x1-x0<20 || y1-y0<20) return {};
  const cv::Rect roi(x0,y0,x1-x0,y1-y0);
  cv::Mat warp=cv::Mat::eye(2,3,CV_32F);
  try {
    const double correlation=cv::findTransformECC(before(roi),after(roi),warp,
        target ? cv::MOTION_AFFINE : cv::MOTION_TRANSLATION,
        {cv::TermCriteria::COUNT|cv::TermCriteria::EPS,40,1e-5});
    if (!std::isfinite(correlation) || correlation<min_correlation || !cv::checkRange(warp)) return {};
    cv::Matx23f transform;
    for (int i=0;i<2;++i) for (int j=0;j<3;++j) transform(i,j)=warp.at<float>(i,j);
    const cv::Matx22f linear(transform(0,0),transform(0,1),transform(1,0),transform(1,1));
    if (target) {
      cv::Mat singular;
      cv::SVD::compute(cv::Mat(linear),singular);
      if (singular.at<float>(0)>1.1F || singular.at<float>(1)<.9F || cv::determinant(linear)<=0) return {};
    }
    const cv::Vec2f origin{float(x0),float(y0)};
    const cv::Vec2f translation=cv::Vec2f(transform(0,2),transform(1,2))+origin-linear*origin;
    transform(0,2)=translation[0]; transform(1,2)=translation[1];
    return transform;
  } catch (const cv::Exception&) { return {}; }
}
cv::Point2f transfer(const cv::Matx23f& transform,const cv::Point2f& p) {
  const auto v=transform*cv::Vec3f(p.x,p.y,1);
  return {v[0],v[1]};
}
bool inside(const cv::Point2f& p,const cv::Size& size) {
  return finite(p) && p.x>=0 && p.y>=0 && p.x<size.width && p.y<size.height;
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
  previous_gray_.release(); previous_detections_.clear(); previous_time_={}; diagnostics_={};
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
  cv::Mat gray;
  cv::cvtColor(frame.bgr,gray,cv::COLOR_BGR2GRAY);
  if (!previous_gray_.empty()) {
    const float alpha=static_cast<float>(-std::expm1(-dt/config_.measurement_time_constant_s));
    for (auto& current : frame.detections) {
      const double size=diameter(current),gate=std::max(2.0,.2*size);
      // R is shared across blades. A target switch must not move the hub to a
      // different NN candidate's regressed R position.
      const Keypoints* hub=nullptr;
      double nearest=std::max(30.0,size);
      if (current.point_valid[4] && finite(current.center_r)) {
        for (const auto& old : previous_detections_) if (old.point_valid[4]) {
          const double distance=cv::norm(current.center_r-old.center_r);
          if (distance<nearest) { nearest=distance; hub=&old; }
        }
        if (hub) {
          const auto warp=registration(previous_gray_,gray,(hub->center_r+current.center_r)*.5F,
                                       40,false,config_.min_correlation);
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
      const auto center=current.armorCenter();
      if (!center || !finite(*center)) continue;
      const Keypoints* target=nullptr;
      nearest=std::max(30.0,.5*size);
      for (const auto& old : previous_detections_) {
        if (old.class_id!=current.class_id || !fullTarget(old)) continue;
        const auto old_center=old.armorCenter();
        if (!old_center) continue;
        const double distance=cv::norm(*center-*old_center);
        if (distance<nearest) { nearest=distance; target=&old; }
      }
      if (!target) continue;
      const auto before=targetPoints(*target),now=targetPoints(current);
      const auto old_center=*target->armorCenter();
      double radius=0;
      for (const auto& p : before) radius=std::max(radius,cv::norm(p-old_center));
      const int half=static_cast<int>(std::clamp(std::ceil(radius+15),20.0,128.0));
      const auto warp=registration(previous_gray_,gray,old_center,half,true,config_.min_correlation);
      if (!warp) continue;
      std::array<cv::Point2f,4> predicted;
      bool valid=true;
      for (int j=0;j<4;++j) {
        predicted[j]=transfer(*warp,before[j]);
        valid &= inside(predicted[j],gray.size()) && cv::norm(predicted[j]-now[j])<gate;
      }
      if (!valid) continue;
      std::array<cv::Point2f*,4> output{&current.top,&current.left,&current.right,&current.bottom};
      for (int j=0;j<4;++j) *output[j]=alpha*now[j]+(1-alpha)*predicted[j];
      ++diagnostics_.refined_targets;
    }
  }
  previous_gray_=std::move(gray); previous_detections_=frame.detections;
  previous_time_=frame.captured_at; previous_mode_=frame.mode; previous_color_=frame.color;
}
} // namespace Rune
