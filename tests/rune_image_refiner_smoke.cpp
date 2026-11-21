#include "rune/f1-observation/image_refiner.hpp"
#include "runtime/rune_runtime.hpp"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
Rune::Frame initial() {
  Rune::Frame frame;
  frame.captured_at=Rune::TimePoint{}+std::chrono::seconds(1);
  frame.color=Rune::Color::Blue;
  cv::Mat noise(360,480,CV_8UC1);
  cv::RNG rng(12345);
  rng.fill(noise,cv::RNG::UNIFORM,0,255);
  cv::GaussianBlur(noise,noise,{13,13},0);
  cv::cvtColor(noise,frame.bgr,cv::COLOR_GRAY2BGR);
  frame.detections.push_back({{220,240},{180,200},{260,200},{220,160},{100,100},0,.95F});
  return frame;
}
Rune::Frame moved(const Rune::Frame& before,double seconds,cv::Point2f shift,cv::Point2f error) {
  auto frame=before;
  frame.captured_at+=std::chrono::duration_cast<Rune::TimePoint::duration>(std::chrono::duration<double>(seconds));
  cv::Mat warped;
  cv::warpAffine(before.bgr,warped,cv::Mat(cv::Matx23f(1,0,shift.x,0,1,shift.y)),before.bgr.size());
  frame.bgr=std::move(warped);
  auto& k=frame.detections[0];
  for (auto* p : {&k.top,&k.left,&k.right,&k.bottom,&k.center_r}) *p+=shift+error;
  return frame;
}
void check() {
  auto config=Rune::ImageRefinerConfig::fromYaml(YAML::Load("enabled: true"));
  Rune::ImageRefiner refiner(config);
  auto before=initial();
  refiner.update(before);
  const cv::Point2f shift(6,-4),error(3,-2);
  auto frame=moved(before,.02,shift,error);
  refiner.update(frame);
  const auto& k=frame.detections[0];
  require(refiner.diagnostics().refined_r==1 && refiner.diagnostics().refined_targets==1,
          "known image translation was not registered");
  require(cv::norm(k.center_r-(before.detections[0].center_r+shift))<1.2 &&
          cv::norm(k.top-(before.detections[0].top+shift))<1.2,
          "image refinement lagged true motion or failed to suppress localization noise");
  require(k.point_valid==before.detections[0].point_valid && k.confidence==.95F,
          "refinement changed observation validity or detection confidence");
  auto missing=frame;
  missing.captured_at+=std::chrono::milliseconds(20);
  missing.detections.clear();
  refiner.update(missing);
  require(missing.detections.empty(),"refinement synthesized a detection on a missing frame");
  auto stale=moved(before,.5,shift,error);
  const auto raw=stale.detections[0];
  refiner.update(stale);
  require(stale.detections[0].top==raw.top && refiner.diagnostics().refined_targets==0,
          "refinement retained history beyond max_gap_s");
  refiner.reset(); before=initial(); refiner.update(before);
  auto weak=moved(before,.02,shift,error);
  weak.detections[0].point_valid[0]=false;
  weak.detections[0].point_valid[4]=false;
  weak.detections[0].point_confidences[0]=weak.detections[0].point_confidences[4]=.1F;
  const auto weak_raw=weak.detections[0];
  refiner.update(weak);
  require(weak.detections[0].top==weak_raw.top && weak.detections[0].center_r==weak_raw.center_r &&
          !weak.detections[0].point_valid[0] && !weak.detections[0].point_valid[4],
          "weak points were promoted or used as image anchors");
  refiner.reset(); before=initial(); refiner.update(before);
  auto blank=moved(before,.02,shift,error);
  blank.bgr.setTo(cv::Scalar::all(0));
  const auto blank_raw=blank.detections[0];
  refiner.update(blank);
  require(blank.detections[0].top==blank_raw.top && blank.detections[0].center_r==blank_raw.center_r &&
          refiner.diagnostics().refined_r==0 && refiner.diagnostics().refined_targets==0,
          "textureless images were accepted as registered measurements");
  refiner.reset(); before=initial(); refiner.update(before);
  auto switched=moved(before,.02,{100,0},error);
  const auto switched_raw=switched.detections[0];
  refiner.update(switched);
  require(switched.detections[0].top==switched_raw.top && refiner.diagnostics().refined_targets==0,
          "a distant new target inherited the previous blade's coordinates");
  for (const char* invalid : {"max_gap_s: 0", "max_gap_s: .nan",
       "measurement_time_constant_s: -1", "min_correlation: 1.1", "min_correlation: .inf"}) {
    bool threw=false;
    try { (void)Rune::ImageRefinerConfig::fromYaml(YAML::Load(invalid)); }
    catch (const std::invalid_argument&) { threw=true; }
    require(threw,"invalid image refinement configuration was accepted");
  }
}

void checkFastMotion() {
  const auto config=Rune::ImageRefinerConfig::fromYaml(YAML::Load(
      "enabled: true\nmeasurement_time_constant_s: 0.16"));
  Rune::ImageRefiner refiner(config);
  auto before=initial();
  refiner.update(before);
  // This displacement exceeds the local convergence range of identity-seeded
  // registration on this texture. It remains within the association gates.
  const cv::Point2f shift(18,-12),error(3,-2);
  auto frame=moved(before,.02,shift,error);
  refiner.update(frame);
  require(refiner.diagnostics().refined_r==1 && refiner.diagnostics().refined_targets==1,
          "fast image motion bypassed coordinate stabilization");
  require(cv::norm(frame.detections[0].center_r-(before.detections[0].center_r+shift))<.8 &&
          cv::norm(frame.detections[0].top-(before.detections[0].top+shift))<.8,
          "stabilization smoothed real motion instead of localization residuals");
  // A stationary scene with alternating regression noise should stabilize at
  // the same physical points without reducing their validity.
  refiner.reset(); before=initial(); refiner.update(before);
  double residual_sum=0;
  for (int i=1;i<=20;++i) {
    const cv::Point2f noise=i%2 ? cv::Point2f(3,-2) : cv::Point2f(-3,2);
    auto stationary=moved(before,.02*i,{0,0},noise);
    refiner.update(stationary);
    residual_sum+=cv::norm(stationary.detections[0].top-before.detections[0].top);
  }
  require(residual_sum/20<.5,"stationary keypoint regression noise remained visible");
  refiner.reset(); before=initial(); refiner.update(before);
  for (int i=1;i<=12;++i) {
    const cv::Point2f movement(3*i,-2*i);
    const cv::Point2f noise=i%2 ? cv::Point2f(3,-2) : cv::Point2f(-3,2);
    auto moving=moved(before,.02*i,movement,noise);
    refiner.update(moving);
    require(refiner.diagnostics().refined_targets==1 &&
            cv::norm(moving.detections[0].top-(before.detections[0].top+movement))<1,
            "continuous real motion accumulated smoothing lag");
  }
}

void checkSmallTarget() {
  Rune::ImageRefinerConfig config;
  config.enabled=true;
  config.measurement_time_constant_s=.16;
  Rune::ImageRefiner refiner(config);
  auto before=initial();
  cv::resize(before.bgr,before.bgr,{},.4,.4,cv::INTER_AREA);
  auto& points=before.detections[0];
  for (auto* p : {&points.top,&points.left,&points.right,&points.bottom,&points.center_r}) *p*=.4F;
  refiner.update(before);
  const cv::Point2f shift(4,-3),error(1.5F,-1);
  auto frame=moved(before,.02,shift,error);
  refiner.update(frame);
  require(refiner.diagnostics().refined_r==1 && refiner.diagnostics().refined_targets==1 &&
          cv::norm(frame.detections[0].top-(points.top+shift))<.6,
          "small image targets lost stabilization or lagged actual displacement");
}

void checkAppearanceChange() {
  Rune::ImageRefinerConfig config;
  config.enabled=true;
  config.measurement_time_constant_s=.16;
  config.min_correlation=1;
  Rune::ImageRefiner refiner(config);
  auto before=initial();
  refiner.update(before);
  const cv::Point2f shift(6,-4);
  auto frame=moved(before,.02,shift,{3,-2});
  // A small local appearance change breaks perfect patch correlation but
  // leaves enough independently checkable image features to measure motion.
  cv::circle(frame.bgr,{220,200},5,cv::Scalar::all(70),cv::FILLED);
  refiner.update(frame);
  require(refiner.diagnostics().refined_targets==1 &&
          cv::norm(frame.detections[0].top-(before.detections[0].top+shift))<1,
          "a local appearance change caused an unfiltered coordinate jump");
}

void checkRingContinuity() {
  Rune::ImageRefinerConfig config;
  config.enabled=true;
  Rune::ImageRefiner refiner(config);
  auto before=initial();
  before.detections[0].target_ring_status=Rune::TargetRingStatus::Verified;
  refiner.update(before);
  const cv::Point2f shift(6,-4);
  for (int i=1;i<=3;++i) {
    auto frame=moved(before,.02*i,shift,{0,0});
    auto& k=frame.detections[0];
    const auto center=*k.armorCenter();
    for (auto* p : {&k.top,&k.left,&k.right,&k.bottom}) *p=center+(*p-center)*.75F;
    const auto raw=k;
    refiner.update(frame);
    require(refiner.diagnostics().rejected_target_rings==1 &&
            k.target_ring_status==Rune::TargetRingStatus::Rejected && !k.armorCenter() &&
            k.top==raw.top,"a nested ring was blended, accepted or became the next reference");
  }
  auto recovered=moved(before,.08,shift,{0,0});
  refiner.update(recovered);
  require(recovered.detections[0].armorCenter() && refiner.diagnostics().refined_targets==1 &&
          refiner.diagnostics().rejected_target_rings==0,
          "a rejected ring prevented recovery to the observed outer ring");
  refiner.reset();refiner.update(before);
  auto zoomed=before;
  zoomed.captured_at+=std::chrono::milliseconds(20);
  const cv::Matx23f warp(1.04F,0,-6,0,1.04F,-8);
  cv::Mat warped;
  cv::warpAffine(before.bgr,warped,cv::Mat(warp),before.bgr.size());
  zoomed.bgr=std::move(warped);
  auto& k=zoomed.detections[0];
  for (auto* p : {&k.top,&k.left,&k.right,&k.bottom,&k.center_r}) {
    const auto q=warp*cv::Vec3f(p->x,p->y,1);*p={q[0],q[1]};
  }
  const auto expected=k;
  refiner.update(zoomed);
  require(refiner.diagnostics().refined_targets==1 && refiner.diagnostics().rejected_target_rings==0 &&
          k.armorCenter() && cv::norm(k.top-expected.top)<1,
          "real image scale change was treated as ring switching");
  auto stale=moved(before,.5,{0,0},{0,0});
  auto& inner=stale.detections[0];
  const auto center=*inner.armorCenter();
  for (auto* p : {&inner.top,&inner.left,&inner.right,&inner.bottom}) *p=center+(*p-center)*.75F;
  refiner.update(stale);
  require(inner.armorCenter() && refiner.diagnostics().rejected_target_rings==0,
          "ring motion check reused expired image history");
}

void markPrecision(Rune::Frame& frame,bool measured) {
  auto& k=frame.detections[0];
  k.target_ring_status=Rune::TargetRingStatus::Verified;
  k.target_ring_coarse_points={k.top,k.left,k.right,k.bottom};
  k.target_ring_subpixel=measured;
  if (measured) {
    const auto center=*k.armorCenter();
    for (auto* p : {&k.top,&k.left,&k.right,&k.bottom}) *p=center+(*p-center)*.9F;
  }
}

void checkPrecisionContinuity() {
  Rune::ImageRefinerConfig config;config.enabled=true;
  Rune::ImageRefiner refiner(config);
  const auto base=initial();
  auto before=base;markPrecision(before,true);refiner.update(before);
  for (int i=1;i<=60;++i) {
    const cv::Point2f shift(float(2*i),float(-i));
    auto frame=moved(base,.02*i,shift,{0,0});
    const bool measured=i<10 && i%3==0;
    markPrecision(frame,measured);
    refiner.update(frame);
    require(refiner.diagnostics().refined_targets==1 && refiner.diagnostics().rejected_target_rings==0,
            "photometric fallback was confused with switching physical rings");
    if (i<=10)
      require(cv::norm(frame.detections[0].top-(before.detections[0].top+shift))<.2,
              "alternating precision/fallback measurements introduced a jump or motion lag");
    if (i==60)
      require(cv::norm(frame.detections[0].top-(base.detections[0].top+shift))<.2,
              "photometric correction survived an unbounded measurement outage");
  }
  refiner.reset();refiner.update(before);
  auto zoomed=base;zoomed.captured_at+=std::chrono::milliseconds(20);
  const cv::Matx23f warp(1.04F,0,-6,0,1.04F,-8);
  cv::Mat warped;cv::warpAffine(base.bgr,warped,cv::Mat(warp),base.bgr.size());
  zoomed.bgr=std::move(warped);
  auto& k=zoomed.detections[0];
  for (auto* p : {&k.top,&k.left,&k.right,&k.bottom,&k.center_r}) {
    const auto q=warp*cv::Vec3f(p->x,p->y,1);*p={q[0],q[1]};
  }
  markPrecision(zoomed,true);const auto expected=k;
  refiner.update(zoomed);
  require(k.armorCenter() && refiner.diagnostics().rejected_target_rings==0 &&
          cv::norm(k.top-expected.top)<.2,"photometric correction lagged real image scaling");
  refiner.reset();refiner.update(before);
  auto inner=moved(base,.02,{2,-1},{0,0});markPrecision(inner,false);
  auto& nested=inner.detections[0];
  const auto center=*nested.armorCenter();
  for (auto* p : {&nested.top,&nested.left,&nested.right,&nested.bottom})
    *p=center+(*p-center)*.75F;
  nested.target_ring_coarse_points={nested.top,nested.left,nested.right,nested.bottom};
  refiner.update(inner);
  require(!nested.armorCenter() && refiner.diagnostics().rejected_target_rings==1,
          "precision continuity hid an actual change to the inner circle");
  auto recovered=moved(base,.04,{4,-2},{0,0});markPrecision(recovered,false);refiner.update(recovered);
  require(recovered.detections[0].armorCenter() &&
          cv::norm(recovered.detections[0].top-(before.detections[0].top+cv::Point2f(4,-2)))<.2,
          "a rejected circle contaminated precision continuity history");
  refiner.reset();auto fresh=base;markPrecision(fresh,false);refiner.update(fresh);
  require(fresh.detections[0].top==base.detections[0].top,"reset retained a previous precision correction");
}

void checkPrecisionSemanticDirections() {
  Rune::ImageRefinerConfig config;config.enabled=true;
  Rune::ImageRefiner ordinary(config),precise(config);
  const auto base=initial();
  auto coarse_before=base,precise_before=base;
  coarse_before.detections[0].target_ring_status=Rune::TargetRingStatus::Verified;
  markPrecision(precise_before,true);
  ordinary.update(coarse_before);precise.update(precise_before);
  for (int i=1;i<=12;++i) {
    auto coarse=base;
    coarse.captured_at+=std::chrono::milliseconds(20*i);
    // The light texture rotates while independent geometric observations stay
    // fixed. Precision must retain the ordinary tracker's semantic directions.
    cv::Mat moved_image;
    const auto warp=cv::getRotationMatrix2D({220,200},3.0*i,1);
    cv::warpAffine(base.bgr,moved_image,warp,base.bgr.size());
    coarse.bgr=std::move(moved_image);
    coarse.detections[0].target_ring_status=Rune::TargetRingStatus::Verified;
    auto fine=coarse;markPrecision(fine,true);
    ordinary.update(coarse);precise.update(fine);
    const auto& a=coarse.detections[0];const auto& b=fine.detections[0];
    require(a.armorCenter() && b.armorCenter(),"rotating light texture invalidated independent geometry");
    const auto u=a.top-a.bottom,v=b.top-b.bottom;
    require(std::abs(u.cross(v))/(cv::norm(u)*cv::norm(v))<1e-4 &&
            cv::norm(*a.armorCenter()-*b.armorCenter())<.05,
            "precision accumulated semantic rotation or center drift from the light texture");
  }
}

void checkScutFeatures() {
  Rune::ImageRefiner refiner(Rune::ImageRefinerConfig::fromYaml(YAML::Load("enabled: true")));
  auto before=initial();
  auto measured=std::make_shared<Rune::DetectedContours>();
  measured->target_ellipse=cv::RotatedRect({220,200},{80,80},0);
  measured->gap_points[0]=cv::Point2f(220,173);
  measured->fan_points={{160,130},{180,140},{170,170},{150,160}};
  measured->fan_point_ids={0,1,2,3};
  before.detections[0].image_contours=measured;
  refiner.update(before);
  const cv::Point2f shift(6,-4),noise(3,-2);
  auto frame=moved(before,.02,shift,noise);
  auto raw=std::make_shared<Rune::DetectedContours>(*measured);
  raw->target_ellipse->center+=shift+noise;
  *raw->gap_points[0]+=shift+noise;
  for (auto& point : raw->fan_points) point+=shift+noise;
  frame.detections[0].image_contours=raw;
  refiner.update(frame);
  const auto& stable=*frame.detections[0].image_contours;
  require(refiner.diagnostics().refined_feature_targets==1 &&
          refiner.diagnostics().refined_fan_points==4,
          "SCUT features bypassed image motion compensation");
  require(cv::norm(stable.target_ellipse->center-(measured->target_ellipse->center+shift))<1.2 &&
          cv::norm(*stable.gap_points[0]-(*measured->gap_points[0]+shift))<1.2 &&
          cv::norm(stable.fan_points[0]-(measured->fan_points[0]+shift))<1.2,
          "SCUT feature stabilization lagged image motion or retained measurement noise");
  require(raw->fan_points[0]==measured->fan_points[0]+shift+noise &&
          raw->target_ellipse->center==measured->target_ellipse->center+shift+noise,
          "SCUT refinement modified the shared raw observation");
  auto stale=moved(before,.5,shift,noise);
  stale.detections[0].image_contours=raw;
  refiner.update(stale);
  require(stale.detections[0].image_contours==raw &&
          refiner.diagnostics().refined_feature_targets==0,
          "SCUT refinement retained stale feature history");
}

class Detector final : public Rune::IKeypointDetector {
public:
  Rune::Keypoints keypoints;
  std::vector<Rune::Keypoints> detect(const cv::Mat&) override { return {keypoints}; }
};
void checkWithoutSerial() {
  runtime::RuneRuntimeConfig config;
  config.image_refiner.enabled=true;
  auto detector=std::make_unique<Detector>();
  auto* source=detector.get();
  runtime::RuneRuntime runtime(config,{},std::move(detector));
  auto before=initial();
  source->keypoints=before.detections[0];
  runtime.detect(before);
  runtime.suspend(); // Live debug path when no serial feedback is available.
  const cv::Point2f shift(6,-4);
  auto frame=moved(before,.02,shift,{3,-2});
  source->keypoints=frame.detections[0];
  runtime.detect(frame);
  require(cv::norm(frame.detections[0].top-(before.detections[0].top+shift))<1.2 &&
          !runtime.lastCommand().found && !runtime.lastCommand().fire,
          "missing serial feedback disabled visual refinement or enabled control");
}
}
int main() {
  try { check(); checkFastMotion(); checkSmallTarget(); checkAppearanceChange(); checkRingContinuity();
    checkPrecisionContinuity(); checkPrecisionSemanticDirections(); checkWithoutSerial();
    checkScutFeatures();
    std::cout<<"rune_image_refiner_smoke passed\n"; return 0; }
  catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
