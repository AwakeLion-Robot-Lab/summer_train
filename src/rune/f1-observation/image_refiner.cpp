#include "rune/f1-observation/image_refiner.hpp"
#include "opencl_ecc.hpp"
#include "ring_reobservation.hpp"
#include "inactive_arm_refiner.hpp"

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
cv::Point2f targetCenter(const Keypoints& k) {
  const auto p=targetPoints(k);
  return (p[0]+p[1]+p[2]+p[3])*.25F;
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
struct RegistrationImages {
  const cv::Mat& before;
  const cv::Mat& after;
  std::vector<cv::Mat>& before_pyramid;
  int& before_levels;
  std::vector<cv::Mat>& after_pyramid;
  int& after_levels;
};

const cv::Size kFlowWindow(21,21);
constexpr int kFlowLevels = 3;

std::optional<cv::Matx23f> opticalFlow(RegistrationImages& images, const cv::Rect& roi, bool target,
                                      const cv::Mat& mask = {}) {
  std::vector<cv::Point2f> points;
  cv::goodFeaturesToTrack(images.before(roi),points,40,.02,4,mask);
  if (points.size()<6) return {};
  if (images.before_levels < 0)
    images.before_levels = cv::buildOpticalFlowPyramid(
        images.before, images.before_pyramid, kFlowWindow, kFlowLevels);
  if (images.after_levels < 0)
    images.after_levels = cv::buildOpticalFlowPyramid(
        images.after, images.after_pyramid, kFlowWindow, kFlowLevels);
  const int levels = std::min(images.before_levels, images.after_levels);
  for (auto& p : points) p+=cv::Point2f(roi.tl());
  std::vector<cv::Point2f> forward,backward;
  std::vector<uchar> forward_valid,backward_valid;
  std::vector<float> errors,backward_errors;
  const cv::TermCriteria criteria(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,.01);
  cv::calcOpticalFlowPyrLK(images.before_pyramid, images.after_pyramid, points, forward,
                          forward_valid, errors, kFlowWindow, levels, criteria);
  cv::calcOpticalFlowPyrLK(images.after_pyramid, images.before_pyramid, forward, backward,
                          backward_valid, backward_errors, kFlowWindow, levels, criteria);
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
using Warp = std::optional<cv::Matx23f>;

std::optional<cv::Matx23f> acceptedTransform(cv::Matx23f transform, double correlation,
    const detail::EccRegion& region, double min_correlation) {
  if (!std::isfinite(correlation) || correlation < min_correlation ||
      !cv::checkRange(cv::Mat(transform, false))) return {};
  const cv::Matx22f linear(transform(0,0),transform(0,1),transform(1,0),transform(1,1));
  if (region.affine) {
    cv::Mat singular;
    cv::SVD::compute(cv::Mat(linear), singular);
    if (singular.at<float>(0)>1.1F || singular.at<float>(1)<.9F || cv::determinant(linear)<=0) return {};
  }
  const cv::Vec2f origin{float(region.roi.x),float(region.roi.y)};
  const cv::Vec2f translation = cv::Vec2f(transform(0,2),transform(1,2)) + origin - linear*origin;
  transform(0,2) = translation[0];
  transform(1,2) = translation[1];
  return transform;
}

class RegistrationBatch {
public:
  RegistrationBatch(RegistrationImages& images, const cv::UMat& before_gpu, const cv::UMat& after_gpu,
                    double min_correlation, ImageRefinerDiagnostics& diagnostics)
      : images_(images), before_gpu_(before_gpu), after_gpu_(after_gpu),
        min_correlation_(min_correlation), diagnostics_(diagnostics) {}

  std::optional<std::size_t> add(cv::Point2f center,int half,bool affine,cv::Point2f displacement,
                                 const Warp& initial_motion = {},int iterations = 40) {
    if (!finite(center) || center.x<0 || center.y<0 ||
        center.x>=images_.before.cols || center.y>=images_.before.rows) return {};
    const int x = cvRound(center.x), y = cvRound(center.y);
    const auto roi = cv::Rect(x-half,y-half,2*half,2*half) & cv::Rect({},images_.before.size());
    if (roi.width<20 || roi.height<20) return {};
    auto initial = initial_motion.value_or(cv::Matx23f::eye());
    if (initial_motion) {
      initial(0,2) += initial(0,0)*roi.x + initial(0,1)*roi.y - roi.x;
      initial(1,2) += initial(1,0)*roi.x + initial(1,1)*roi.y - roi.y;
    }
    jobs_.push_back({{roi, initial, affine, iterations}, displacement, !initial_motion});
    return jobs_.size() - 1;
  }

  void run() {
    results_.assign(jobs_.size(), {});
    std::vector<detail::EccRegion> regions;
    for (const auto& job : jobs_) regions.push_back(job.region);
    detail::EccPreparedBatch prepared;
    const auto first = estimate(regions, prepared);
    regions.clear();
    std::vector<std::size_t> retries;
    for (std::size_t i = 0; i < jobs_.size(); ++i) {
      results_[i] = first[i];
      const auto& job = jobs_[i];
      if (results_[i] || !job.retry || !finite(job.displacement) || cv::norm(job.displacement)<2) continue;
      auto region = job.region;
      region.initial(0,2) = job.displacement.x;
      region.initial(1,2) = job.displacement.y;
      regions.push_back(region);
      retries.push_back(i);
    }
    const auto second = estimate(regions, prepared, retries);
    for (std::size_t i = 0; i < retries.size(); ++i) results_[retries[i]] = second[i];
    // The existing motion check uses forward/backward CPU optical flow when ECC
    // has insufficient image evidence and there is no verified body motion.
    for (std::size_t i = 0; i < jobs_.size(); ++i) {
      const auto& job = jobs_[i];
      if (!results_[i] && job.retry)
        results_[i] = opticalFlow(images_, job.region.roi, job.region.affine);
    }
    jobs_.clear();
  }

  Warp result(std::optional<std::size_t> index) const {
    return index ? results_[*index] : Warp{};
  }

private:
  struct Job {
    detail::EccRegion region;
    cv::Point2f displacement;
    bool retry;
  };

  std::vector<Warp> estimate(const std::vector<detail::EccRegion>& regions,
                             detail::EccPreparedBatch& prepared,
                             std::span<const std::size_t> retry_indices = {}) {
    std::vector<Warp> results;
    results.reserve(regions.size());
    if (regions.empty()) return results;
    if (!after_gpu_.empty()) {
      const auto measured = detail::estimateOpenCLEcc(before_gpu_, after_gpu_, regions,
                                                     prepared, retry_indices);
      ++diagnostics_.gpu_ecc_batches;
      for (std::size_t i = 0; i < regions.size(); ++i) {
        results.push_back(acceptedTransform(measured[i].transform, measured[i].correlation,
                                           regions[i], min_correlation_));
        diagnostics_.gpu_ecc_accepted += results.back().has_value();
      }
    } else {
      for (const auto& r : regions) {
        auto transform = r.initial;
        cv::Mat warp(transform, false);
        try {
          const double correlation = cv::findTransformECC(images_.before(r.roi), images_.after(r.roi), warp,
              r.affine ? cv::MOTION_AFFINE : cv::MOTION_TRANSLATION,
              {cv::TermCriteria::COUNT|cv::TermCriteria::EPS, r.iterations, 1e-5});
          results.push_back(acceptedTransform(transform, correlation, r, min_correlation_));
        } catch (const cv::Exception& e) {
          if (e.code != cv::Error::StsNoConv) throw;
          results.emplace_back();
        }
      }
    }
    return results;
  }

  RegistrationImages& images_;
  const cv::UMat& before_gpu_;
  const cv::UMat& after_gpu_;
  double min_correlation_;
  ImageRefinerDiagnostics& diagnostics_;
  std::vector<Job> jobs_;
  std::vector<Warp> results_;
};
cv::Point2f transfer(const cv::Matx23f& transform,const cv::Point2f& p) {
  const auto v=transform*cv::Vec3f(p.x,p.y,1);
  return {v[0],v[1]};
}

void validate(const ImageRefinerConfig& config) {
  if (!std::isfinite(config.max_gap_s) || config.max_gap_s<=0 || config.max_gap_s>1 ||
      !std::isfinite(config.measurement_time_constant_s) || config.measurement_time_constant_s<=0 ||
      !std::isfinite(config.min_correlation) || config.min_correlation<=0 || config.min_correlation>1)
    throw std::invalid_argument("invalid rune image refiner configuration");
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

cv::RotatedRect transferEllipse(const cv::Matx23f& warp, const cv::RotatedRect& ellipse) {
  const cv::Matx22f linear(warp(0,0),warp(0,1),warp(1,0),warp(1,1));
  return ellipseFromShape(transfer(warp,ellipse.center),linear*ellipseShape(ellipse)*linear.t());
}

Warp sharedBladeMotion(cv::Point2f before, cv::Point2f before_hub, cv::Point2f after,
                      cv::Point2f after_hub) {
  const auto a=before-before_hub, b=after-after_hub;
  const double length=a.dot(a);
  if (length<25 || cv::norm(after_hub-before_hub)>std::max(15.0,.15*std::sqrt(length))) return {};
  const float c=float(a.dot(b)/length), s=float(a.cross(b)/length);
  const float scale=std::hypot(c,s);
  if (scale<.85F || scale>1.18F || std::abs(std::atan2(s,c))>.28F) return {};
  return cv::Matx23f(c,-s,after_hub.x-c*before_hub.x+s*before_hub.y,
                    s,c,after_hub.y-s*before_hub.x-c*before_hub.y);
}

bool hasColoredRim(const cv::Mat& bgr, const cv::RotatedRect& ellipse, Color color) {
  const double angle=ellipse.angle*CV_PI/180, c=std::cos(angle), s=std::sin(angle);
  int supported=0, quadrants=0;
  for (int quadrant=0; quadrant<4; ++quadrant) {
    int local=0;
    for (int i=0; i<8; ++i) {
      const double t=(quadrant*8+i)*2*CV_PI/32;
      const double x=.5*ellipse.size.width*std::cos(t), y=.5*ellipse.size.height*std::sin(t);
      const cv::Point2f radial(float(c*x-s*y),float(s*x+c*y));
      for (const float ratio : {.88F,.95F,1.0F,1.05F,1.12F}) {
        const cv::Point p=ellipse.center+ratio*radial;
        if (p.x<0 || p.y<0 || p.x>=bgr.cols || p.y>=bgr.rows) continue;
        const auto pixel=bgr.at<cv::Vec3b>(p);
        const int primary=color==Color::Blue ? pixel[0] : pixel[2];
        const int opposite=color==Color::Blue ? pixel[2] : pixel[0];
        if (primary-opposite>=std::max(14,primary/8)) { ++local; break; }
      }
    }
    supported+=local;
    quadrants+=local>=2;
  }
  return supported>=8 && quadrants>=3;
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

struct FanRegistration {
  struct Pair { std::size_t current, previous; };
  std::vector<Pair> pairs;
  std::optional<std::size_t> request;
  Warp body_motion;
};

FanRegistration scheduleFanRegistration(const DetectedContours& current,
    const DetectedContours& raw_previous, RegistrationBatch& batch, const Warp& body_motion) {
  FanRegistration result;
  result.body_motion = body_motion;
  cv::Point2f center{};
  std::vector<float> dx,dy;
  for (std::size_t i=0;i<current.fan_points.size();++i) {
    if (!current.fanPointValid(i) || !finite(current.fan_points[i])) continue;
    for (std::size_t j=0;j<raw_previous.fan_points.size();++j) {
      if (raw_previous.fanPointId(j)!=current.fanPointId(i) || !raw_previous.fanPointValid(j) ||
          !finite(raw_previous.fan_points[j])) continue;
      const auto shift=current.fan_points[i]-raw_previous.fan_points[j];
      result.pairs.push_back({i,j}); center+=raw_previous.fan_points[j];
      dx.push_back(shift.x);dy.push_back(shift.y);break;
    }
  }
  if (result.pairs.size() >= 3) {
    center*=1.F/result.pairs.size();
    const auto median=[](std::vector<float>& values) {
      const auto middle=values.begin()+values.size()/2;
      std::nth_element(values.begin(),middle,values.end());return *middle;
    };
    const cv::Point2f displacement(median(dx),median(dy));
    double radius=0;
    for (const auto& pair : result.pairs)
      radius=std::max(radius,cv::norm(raw_previous.fan_points[pair.previous]-center));
    const int half=int(std::clamp(std::ceil(radius+15),20.0,256.0));
    // 灯臂单独配准。光斑图案变化较大时，保留经图像核验的叶片运动。
    result.request = batch.add(center,half,true,displacement,body_motion,20);
  }
  return result;
}

void stabilizeFanFeatures(DetectedContours& out,const DetectedContours& previous,
    const FanRegistration& fan, const RegistrationBatch& batch, cv::Size image_size,
    float alpha,double target_diameter,ImageRefinerDiagnostics& diagnostics) {
  if (fan.pairs.empty()) return;
  auto warp = batch.result(fan.request);
  if (!warp) warp = fan.body_motion;
  if (!warp) return;
  if (out.fan_point_valid.size() != out.fan_points.size())
    out.fan_point_valid.assign(out.fan_points.size(),true);
  const double gate=std::max(2.0,.12*target_diameter);
  for (const auto& pair : fan.pairs) {
    const int id = out.fanPointId(pair.current);
    for (std::size_t j=0;j<previous.fan_points.size();++j) {
      if (previous.fanPointId(j)!=id) continue;
      const bool previously_measured = previous.fanPointValid(j);
      const auto prior = previously_measured ? std::optional{previous.fan_points[j]}
                                             : previous.fanDisplayPrediction(j);
      if (!prior) break;
      const auto predicted=transfer(*warp,*prior);
      const bool in_image = inside(predicted,image_size);
      if (!in_image || cv::norm(predicted-out.fan_points[pair.current])>gate) {
        out.fan_point_valid[pair.current]=false;++diagnostics.rejected_feature_points;
        // 只有来自上一帧有效观测的点可以产生一帧显示预测。
        if (in_image && previously_measured && id >= 0 &&
            std::size_t(id) < out.fan_display_predictions.size())
          out.fan_display_predictions[id] = predicted;
      } else {
        out.fan_points[pair.current]=alpha*out.fan_points[pair.current]+(1-alpha)*predicted;
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
    const auto backend = node["registration_backend"].as<std::string>("cpu");
    if (backend != "cpu" && backend != "opencl")
      throw std::invalid_argument("rune.image_refiner.registration_backend must be cpu or opencl");
    config.use_opencl = backend == "opencl";
    config.opencl_device = node["opencl_device"].as<std::string>("");
    config.track_numbered_points = node["track_numbered_points"].as<bool>(config.track_numbered_points);
    config.reobserve_target_rings = node["reobserve_target_rings"].as<bool>(config.reobserve_target_rings);
    config.refine_inactive_arm_edges =
        node["refine_inactive_arm_edges"].as<bool>(config.refine_inactive_arm_edges);
  }
  validate(config);
  return config;
}
ImageRefiner::ImageRefiner(ImageRefinerConfig config) : config_(std::move(config)) {
  validate(config_);
  if (!config_.enabled || !config_.use_opencl) return;
  const auto current = cv::ocl::OpenCLExecutionContext::getCurrent();
  if (config_.opencl_device.empty()) {
    opencl_ = current;
  } else {
    const auto context = cv::ocl::Context::create(config_.opencl_device);
    if (context.ndevices() != 1)
      throw std::runtime_error("cannot select rune OpenCL device: " + config_.opencl_device);
    opencl_ = cv::ocl::OpenCLExecutionContext::create(context, context.device(0));
  }
  if (opencl_.empty() || !opencl_.useOpenCL() ||
      (opencl_.getDevice().type() & cv::ocl::Device::TYPE_GPU) == 0)
    throw std::runtime_error("rune OpenCL registration requires an active OpenCV OpenCL GPU");
  const cv::ocl::OpenCLExecutionContextScope scope(opencl_);
  detail::prepareOpenCLEcc();
}

std::string ImageRefiner::registrationDevice() const {
  return opencl_.empty() ? "CPU" : opencl_.getDevice().name();
}

void ImageRefiner::updateDisplayOutlines(Frame& frame, const cv::Mat& gray) {
  std::vector<OutlineTrack> measured;
  for (const auto& detection : frame.detections) {
    const auto& features=detection.image_contours;
    if (detection.target_ring_status!=TargetRingStatus::Verified || !features || !features->target_ellipse)
      continue;
    const auto& ellipse=*features->target_ellipse;
    if (!inside(ellipse.center,gray.size()) || !std::isfinite(ellipse.angle) ||
        !std::isfinite(ellipse.size.width) || !std::isfinite(ellipse.size.height) ||
        std::min(ellipse.size.width,ellipse.size.height)<4) continue;
    OutlineTrack track;
    track.shape.target=ellipse;
    track.shape.class_id=detection.class_id;
    track.shape.arm_points=features->fan_display_predictions;
    for (std::size_t j=0;j<features->fan_points.size();++j) {
      const int id=features->fanPointId(j);
      if (id>=0 && std::size_t(id)<track.shape.arm_points.size() && features->fanPointValid(j))
        track.shape.arm_points[id]=features->fan_points[j];
    }
    if (const auto& outline=features->inactive_arm_outline) {
      for (std::size_t i=0;i<outline->corners.size();++i)
        track.shape.arm_points[i]=outline->corners[i];
    }
    if (detection.point_valid[4] && inside(detection.center_r,gray.size())) track.hub=detection.center_r;
    track.observed_at=frame.captured_at;
    measured.push_back(std::move(track));
  }
  const double dt=std::chrono::duration<double>(frame.captured_at-outline_time_).count();
  if (!outline_tracks_.empty() && dt>0 && dt<=config_.max_gap_s) {
    struct Match { std::size_t old_index,new_index; double distance; };
    std::vector<Match> candidates;
    for (std::size_t i=0;i<outline_tracks_.size();++i) for (std::size_t j=0;j<measured.size();++j) {
      const auto& before=outline_tracks_[i].shape.target;
      const auto& after=measured[j].shape.target;
      const auto distance=cv::norm(before.center-after.center);
      const double size=std::max(before.size.width,before.size.height);
      if (distance<=std::max(25.0,.55*size)) candidates.push_back({i,j,distance});
    }
    std::sort(candidates.begin(),candidates.end(),[](const Match& a,const Match& b) {
      return a.distance<b.distance;
    });
    std::vector<bool> matched(outline_tracks_.size()), used(measured.size());
    std::vector<cv::Matx23f> observed_motions;
    for (const auto& [i,j,distance] : candidates) {
      (void)distance;
      if (matched[i] || used[j]) continue;
      matched[i]=used[j]=true;
      const auto& before=outline_tracks_[i];
      const auto& after=measured[j];
      if (before.observed_at!=outline_time_ || !before.hub || !after.hub) continue;
      if (auto warp=sharedBladeMotion(before.shape.target.center,*before.hub,
                                     after.shape.target.center,*after.hub))
        observed_motions.push_back(*warp);
    }

    const std::size_t measurements=measured.size();
    std::vector<cv::Mat> old_pyramid, new_pyramid;
    int old_levels=-1, new_levels=-1;
    RegistrationImages images{outline_gray_,gray,old_pyramid,old_levels,new_pyramid,new_levels};
    for (std::size_t i=0;i<outline_tracks_.size();++i) {
      auto track=outline_tracks_[i];
      const double age=std::chrono::duration<double>(frame.captured_at-track.observed_at).count();
      if (matched[i] || age>config_.max_gap_s) continue;
      const auto& old=track.shape.target;
      const double diameter=std::max(old.size.width,old.size.height);
      const auto reobserve=[&](const cv::RotatedRect& search,bool shared,bool history) {
        if (!config_.reobserve_target_rings) return;
        const auto start=std::chrono::steady_clock::now();
        if (auto ring=detail::reobserveRing(frame.bgr,frame.color,search,shared)) {
          const auto overlaps=[&](const cv::RotatedRect& other) {
            return cv::norm(ring->ellipse.center-other.center)<.45*diameter;
          };
          const bool duplicate=std::any_of(measured.begin(),measured.begin()+measurements,
              [&](const auto& current) { return overlaps(current.shape.target); }) ||
              std::any_of(frame.reobserved_target_rings.begin(),frame.reobserved_target_rings.end(),
              [&](const auto& current) { return overlaps(current.ellipse); });
          if (!duplicate) {
            ring->from_history=history;
            frame.reobserved_target_rings.push_back(*ring);
          }
        }
        diagnostics_.ring_reobservation_ms+=
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      };
      Warp warp;
      if (!track.history_only && !observed_motions.empty()) {
        const auto center=transfer(observed_motions.front(),old.center);
        if (std::all_of(observed_motions.begin()+1,observed_motions.end(),[&](const auto& w) {
              return cv::norm(center-transfer(w,old.center))<=std::max(3.0,.1*diameter);
            })) warp=observed_motions.front();
      }
      const bool shared_motion=warp.has_value();
      if (!shared_motion && !track.history_only) {
        const cv::Point center=old.center;
        const int half=std::clamp(cvCeil(.65*diameter),12,128);
        const auto roi=cv::Rect(center.x-half,center.y-half,2*half,2*half)&cv::Rect({},gray.size());
        if (roi.width>=20 && roi.height>=20) {
          cv::Mat mask=cv::Mat::zeros(roi.size(),CV_8U);
          auto local=old;
          local.center-=cv::Point2f(roi.tl());
          cv::ellipse(mask,local,cv::Scalar(255),cv::FILLED);
          warp=opticalFlow(images,roi,true,mask);
        }
      }
      if (!warp) {
        if (config_.reobserve_target_rings) {
          reobserve(old,false,true);
          track.history_only=true;
          measured.push_back(std::move(track));
        }
        continue;
      }
      const auto predicted=transferEllipse(*warp,old);
      if (!inside(predicted.center,gray.size()) ||
          (!shared_motion && !hasColoredRim(frame.bgr,predicted,frame.color)) ||
          std::any_of(measured.begin(),measured.begin()+measurements,[&](const auto& current) {
            return cv::norm(predicted.center-current.shape.target.center)<.45*diameter;
          })) continue;
      track.shape.target=predicted;
      for (auto& p : track.shape.arm_points) if (p) {
        const auto next=transfer(*warp,*p);
        p=inside(next,gray.size()) ? std::optional{next} : std::nullopt;
      }
      if (track.hub) track.hub=transfer(*warp,*track.hub);
      reobserve(predicted,shared_motion,false);
      frame.display_outline_predictions.push_back(track.shape);
      measured.push_back(std::move(track));
    }
  }
  outline_tracks_=std::move(measured);
  outline_gray_=outline_tracks_.empty() ? cv::Mat{} : gray;
  outline_time_=frame.captured_at;
}

void ImageRefiner::reset() {
  previous_gray_.release(); previous_gpu_gray_.release();
  previous_flow_pyramid_.clear(); previous_flow_levels_=-1;
  previous_detections_.clear(); previous_raw_detections_.clear();
  numbered_point_tracks_.clear();
  outline_tracks_.clear(); outline_gray_.release(); outline_time_={};
  previous_time_={}; diagnostics_={};
}
void ImageRefiner::update(Frame& frame) {
  diagnostics_={};
  frame.display_outline_predictions.clear();
  frame.reobserved_target_rings.clear();
  if (!config_.enabled) { reset(); return; }
  if (frame.bgr.empty() || frame.bgr.type()!=CV_8UC3 || frame.captured_at==TimePoint{}) { reset(); return; }
  const double dt=std::chrono::duration<double>(frame.captured_at-previous_time_).count();
  if (dt<=0 || dt>config_.max_gap_s || frame.bgr.size()!=previous_gray_.size() ||
      frame.mode!=previous_mode_ || frame.color!=previous_color_) reset();
  // No detection means no observed measurement: retain only bounded history.
  if (frame.detections.empty()) {
    if (!outline_tracks_.empty()) {
      cv::Mat gray;
      cv::cvtColor(frame.bgr,gray,cv::COLOR_BGR2GRAY);
      updateDisplayOutlines(frame,gray);
    }
    return;
  }
  auto raw_detections=frame.detections;
  std::vector<int> feature_associations(frame.detections.size(),-1);
  std::vector<std::shared_ptr<DetectedContours>> refined_contours;
  if (config_.track_numbered_points) refined_contours.resize(frame.detections.size());
  cv::Mat gray;
  cv::cvtColor(frame.bgr,gray,cv::COLOR_BGR2GRAY);
  std::vector<cv::Mat> flow_pyramid;
  int flow_levels = -1;
  cv::UMat gpu_gray;
  std::optional<cv::ocl::OpenCLExecutionContextScope> gpu_scope;
  if (!opencl_.empty()) {
    gpu_scope.emplace(opencl_);
    gray.copyTo(gpu_gray);
  }
  if (!previous_gray_.empty()) {
    RegistrationImages images{previous_gray_, gray, previous_flow_pyramid_, previous_flow_levels_,
                              flow_pyramid, flow_levels};
    RegistrationBatch batch(images,previous_gpu_gray_,gpu_gray,config_.min_correlation,diagnostics_);
    const float alpha=static_cast<float>(-std::expm1(-dt/config_.measurement_time_constant_s));
    std::vector<Warp> feature_warps(frame.detections.size());
    struct HubRegistration {
      cv::Point2f previous, current;
      std::optional<std::size_t> request;
    };
    struct Association {
      const Keypoints* hub{nullptr};
      std::optional<std::size_t> hub_request;
      std::optional<std::size_t> target_request;
    };
    std::vector<HubRegistration> hub_registrations;
    std::vector<Association> associations(frame.detections.size());
    for (std::size_t index=0;index<frame.detections.size();++index) {
      const auto& current=frame.detections[index];
      auto& association=associations[index];
      const double size=diameter(current);
      // 多片叶子共享 R 标，切换叶片时继续关联同一 R 标。
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
          auto cached = std::find_if(hub_registrations.begin(), hub_registrations.end(),
              [&](const auto& item) {
                return item.previous == raw_hub->center_r && item.current == current.center_r;
              });
          if (cached == hub_registrations.end()) {
            const auto midpoint = (raw_hub->center_r + current.center_r) * .5F;
            hub_registrations.push_back({raw_hub->center_r, current.center_r,
                batch.add(midpoint,40,false,current.center_r-raw_hub->center_r)});
            cached = hub_registrations.end() - 1;
            ++diagnostics_.hub_registrations;
          }
          association.hub = hub;
          association.hub_request = cached->request;
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
      const auto old_center=targetCenter(*raw_target);
      double radius=0;
      for (const auto& p : targetPoints(*raw_target)) radius=std::max(radius,cv::norm(p-old_center));
      const int half=static_cast<int>(std::clamp(std::ceil(radius+15),20.0,128.0));
      association.target_request = batch.add(old_center,half,true,center-old_center);
    }
    batch.run();
    for (std::size_t index=0;index<frame.detections.size();++index) {
      auto& current=frame.detections[index];
      const auto& association = associations[index];
      const double size=diameter(current),gate=std::max(2.0,.2*size);
      if (const auto hub_warp = batch.result(association.hub_request)) {
        const auto p=transfer(*hub_warp,association.hub->center_r);
        if (inside(p,gray.size()) && cv::norm(p-current.center_r)<gate) {
          current.center_r=alpha*current.center_r+(1-alpha)*p;
          ++diagnostics_.refined_r;
        }
      }
      const auto warp = batch.result(association.target_request);
      feature_warps[index]=warp;
      if (!warp) continue;
      const auto old_index=std::size_t(feature_associations[index]);
      const auto* target=&previous_detections_[old_index];
      const auto before=targetPoints(*target);
      const auto now=targetPoints(current);
      std::array<cv::Point2f,4> predicted;
      bool valid=true;
      for (int j=0;j<4;++j) {
        predicted[j]=transfer(*warp,before[j]);
        valid &= inside(predicted[j],gray.size()) &&
            cv::norm(predicted[j]-now[j])<gate;
      }
      if (current.target_ring_status==TargetRingStatus::Verified &&
          target->target_ring_status==TargetRingStatus::Verified) {
        // Compare diameters AFTER independently measuring image motion. Real
        // approach, rotation and foreshortening move both predictions together;
        // switching to a nested ring changes the measured diameters alone.
        bool consistent=valid;
        for (const auto [a,b] : {std::pair{0,3},std::pair{1,2}}) {
          const double length=cv::norm(predicted[a]-predicted[b]);
          consistent &= std::abs(cv::norm(now[a]-now[b])-length)<=std::max(2.0,.08*length);
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
        *output[j]=alpha*now[j]+(1-alpha)*predicted[j];
      ++diagnostics_.refined_targets;
    }
    std::vector<FanRegistration> fans(frame.detections.size());
    for (std::size_t i=0;i<fans.size();++i) {
      const auto& current=frame.detections[i].image_contours;
      const int old_index=feature_associations[i];
      if (!current || old_index<0) continue;
      const auto& old=previous_detections_[old_index].image_contours;
      const auto& raw_old=previous_raw_detections_[old_index].image_contours;
      if (!old || !raw_old || old->fan_points.empty()) continue;
      fans[i]=scheduleFanRegistration(*current,*raw_old,batch,
          bladeImageMotion(previous_raw_detections_[old_index],raw_detections[i],feature_warps[i]));
    }
    batch.run();
    for (std::size_t i=0;i<frame.detections.size();++i) {
      auto& current=frame.detections[i];
      if (!current.image_contours) continue;
      // Shared raw contour data is immutable. Never filter in place: the raw
      // association history and callers may still own this exact measurement.
      auto features=std::make_shared<DetectedContours>(*current.image_contours);
      features->fan_display_predictions = {};
      if (current.target_ring_status==TargetRingStatus::Rejected) {
        features->target_ellipse.reset();features->gap_points={};
      }
      const int old_index=feature_associations[i];
      if (old_index>=0) {
        const auto& old=previous_detections_[old_index].image_contours;
        const auto& raw_old=previous_raw_detections_[old_index].image_contours;
        if (old && raw_old) {
          if (feature_warps[i]) stabilizeTargetFeatures(*features,*old,*feature_warps[i],alpha,
                                                        gray.size(),diagnostics_);
          stabilizeFanFeatures(*features,*old,fans[i],batch,gray.size(),alpha,
                               diameter(raw_detections[i]),diagnostics_);
        }
      }
      if (current.target_ring_status == TargetRingStatus::Rejected)
        features->fan_display_predictions = {};
      if (config_.track_numbered_points) refined_contours[i]=features;
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
  if (rejected && !observed_target) {
    if (config_.track_numbered_points || config_.reobserve_target_rings) updateDisplayOutlines(frame,gray);
    return;
  }
  std::vector<unsigned> association_counts;
  if (config_.track_numbered_points || config_.refine_inactive_arm_edges) {
    association_counts.resize(previous_detections_.size());
    for (int index : feature_associations)
      if (index>=0) ++association_counts[index];
  }
  if (config_.track_numbered_points) {
    std::vector<NumberedPointEkf> next_tracks(frame.detections.size());
    for (std::size_t i=0; i<frame.detections.size(); ++i) {
      auto& blade=frame.detections[i];
      if (!blade.image_contours || blade.target_ring_status==TargetRingStatus::Rejected) continue;
      const int index=feature_associations[i];
      if (index>=0 && std::size_t(index)<numbered_point_tracks_.size() && association_counts[index]==1)
        next_tracks[i]=numbered_point_tracks_[index];
      auto output=std::move(refined_contours[i]);
      if (!output) output=std::make_shared<DetectedContours>(*blade.image_contours);
      next_tracks[i].update(blade,*output,gray.size(),frame.captured_at,
          config_.max_gap_s,config_.measurement_time_constant_s);
      blade.image_contours=std::move(output);
    }
    numbered_point_tracks_=std::move(next_tracks);
  }
  if (config_.refine_inactive_arm_edges) {
    const auto start=std::chrono::steady_clock::now();
    const float weight=static_cast<float>(-std::expm1(-dt/config_.measurement_time_constant_s));
    for (std::size_t i=0;i<frame.detections.size();++i) {
      auto& blade=frame.detections[i];
      if (blade.class_id!=0 || !blade.image_contours ||
          blade.image_contours->fan_points.size()!=4 ||
          blade.target_ring_status!=TargetRingStatus::Verified) continue;
      const int index=feature_associations[i];
      const Keypoints* prior=index>=0 && association_counts[index]==1
          ? &previous_detections_[index] : nullptr;
      auto output=std::make_shared<DetectedContours>(*blade.image_contours);
      detail::refineInactiveArm(frame.bgr,frame.color,blade,raw_detections[i],prior,
                                frame.captured_at,config_.max_gap_s,weight,*output);
      blade.image_contours=std::move(output);
    }
    diagnostics_.inactive_arm_ms=
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  }
  if (config_.track_numbered_points || config_.reobserve_target_rings) updateDisplayOutlines(frame,gray);
  previous_gray_=std::move(gray); previous_gpu_gray_=std::move(gpu_gray);
  previous_flow_pyramid_=std::move(flow_pyramid); previous_flow_levels_=flow_levels;
  previous_detections_=frame.detections;
  previous_raw_detections_=std::move(raw_detections);
  previous_time_=frame.captured_at; previous_mode_=frame.mode; previous_color_=frame.color;
}
} // namespace Rune
