#include "rune/f1-observation/target_ring_refiner.hpp"
#include "rune/f1-observation/keypoint_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <array>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace {
constexpr double ratio=60.0/280;
const cv::Point2f center(400,350);
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
Rune::Keypoints points(float scale=1) {
  return {{400,350+60*scale},{400-60*scale,350},{400+60*scale,350},
          {400,350-60*scale},{400,70},0,.95F};
}
std::array<cv::Point2f,4> target(const Rune::Keypoints& k) { return {k.top,k.left,k.right,k.bottom}; }
cv::Mat image(bool outer=true,cv::Scalar color={255,30,10}) {
  cv::Mat bgr(600,800,CV_8UC3,cv::Scalar::all(8));
  for (int radius : {27,42,60}) {
    if (!outer && radius==60) continue;
    // Individual LEDs, including concentric rings and radial spokes.
    for (int i=0;i<72;++i) {
      const double angle=i*2*CV_PI/72;
      cv::circle(bgr,center+cv::Point2f(float(radius*std::cos(angle)),float(radius*std::sin(angle))),
                 2,color,cv::FILLED,cv::LINE_AA);
    }
  }
  return bgr;
}
void checkSwitching() {
  const auto bgr=image();
  auto reference=points();
  require(Rune::refineTargetRing(bgr,reference,Rune::Mode::Small,ratio)==Rune::TargetRingResult::Refined,
          "outer LED ring was not identified");
  for (int i=0;i<30;++i) {
    auto k=points(i%2 ? .45F : 1);
    if (i%3==0) { k.top=center+(k.top-center)*.6F; k.bottom=center+(k.bottom-center)*.6F; }
    const auto raw=k;
    require(Rune::refineTargetRing(bgr,k,Rune::Mode::Small,ratio)==Rune::TargetRingResult::Refined,
            "changing NN radius changed ring acceptance");
    const auto refined=target(k),expected=target(reference);
    for (int j=0;j<4;++j) {
      require(cv::norm(refined[j]-expected[j])<.01,"inner/outer NN switching leaked to output");
      require(std::abs(cv::norm(refined[j]-center)-60)<3,"target point missed the outer ring");
    }
    require(k.center_r==raw.center_r && k.point_valid==raw.point_valid &&
            k.point_confidences==raw.point_confidences && k.confidence==raw.confidence &&
            k.class_id==raw.class_id,"ring correction changed R, scores or class");
  }
}
cv::Point2f transform(const cv::Matx23f& warp,cv::Point2f p) {
  const auto q=warp*cv::Vec3f(p.x,p.y,1);return {q[0],q[1]};
}
void checkMotion() {
  const auto bgr=image();
  auto reference=points();
  (void)Rune::refineTargetRing(bgr,reference,Rune::Mode::Small,ratio);
  for (int i=1;i<12;++i) {
    const float angle=.04F*i,scale=1+.005F*i;
    const float c=scale*std::cos(angle),s=scale*std::sin(angle);
    const cv::Matx23f warp(c,-s,400-c*400+s*350+2*i,s,c,350-s*400-c*350-i);
    cv::Mat moved;cv::warpAffine(bgr,moved,cv::Mat(warp),bgr.size());
    auto k=points(i%2 ? .45F : 1);
    for (auto* p : {&k.top,&k.left,&k.right,&k.bottom,&k.center_r}) *p=transform(warp,*p);
    require(Rune::refineTargetRing(moved,k,Rune::Mode::Small,ratio)==Rune::TargetRingResult::Refined,
            "real translation/rotation/scale was rejected");
    const auto actual=target(k),before=target(reference);
    for (int j=0;j<4;++j)
      require(cv::norm(actual[j]-transform(warp,before[j]))<2,"ring correction lagged real motion");
  }
  // An inclined circular plane projects to an ellipse, with oblique image axes.
  const cv::Matx23f warp(.7F,.15F,80,0,1,0);
  cv::Mat inclined;cv::warpAffine(bgr,inclined,cv::Mat(warp),bgr.size());
  auto k=points(.45F);
  for (auto* p : {&k.top,&k.left,&k.right,&k.bottom,&k.center_r}) *p=transform(warp,*p);
  require(Rune::refineTargetRing(inclined,k,Rune::Mode::Small,ratio)==Rune::TargetRingResult::Refined,
          "foreshortened ring was rejected");
  const auto actual=target(k),before=target(reference);
  for (int j=0;j<4;++j)
    require(cv::norm(actual[j]-transform(warp,before[j]))<2,"ellipse correction changed semantic directions");
}
void checkDisconnectedLeds() {
  for (const cv::Scalar color : {cv::Scalar(255,30,10),cv::Scalar(10,30,255)}) {
    cv::Mat bgr(600,800,CV_8UC3,cv::Scalar::all(8));
    // Sparse LEDs leave a complete physical rim visible, but split the ordinary
    // closed mask into independent arcs, as in the live capture.
    for (int radius : {27,42,60}) for (int i=0;i<40;++i) {
      const double angle=i*2*CV_PI/40;
      cv::circle(bgr,center+cv::Point2f(float(radius*std::cos(angle)),float(radius*std::sin(angle))),
                 2,color,cv::FILLED,cv::LINE_AA);
    }
    for (const bool subpixel : {false,true}) for (const float scale : {.45F,1.0F}) {
      auto k=points(scale);
      const auto raw=k;
      Rune::TargetRingDiagnostics diagnostics;
      require(Rune::refineTargetRing(bgr,k,Rune::Mode::Small,ratio,subpixel,&diagnostics)==
                Rune::TargetRingResult::Refined && diagnostics.reconnected &&
                diagnostics.rejection==Rune::TargetRingRejectReason::None,
              "short LED gaps were not recovered by the bounded retry");
      for (const auto& p : target(k))
        require(std::abs(cv::norm(p-center)-60)<3,"LED-gap recovery selected an inner ring");
      require(k.point_valid==raw.point_valid && k.center_r==raw.center_r &&
              k.point_confidences==raw.point_confidences,"LED-gap recovery changed detection reliability");
    }
  }
  auto k=points();
  Rune::TargetRingDiagnostics diagnostics;
  require(Rune::refineTargetRing(cv::Mat::zeros(600,800,CV_8UC3),k,Rune::Mode::Small,
                                ratio,true,&diagnostics)==Rune::TargetRingResult::Rejected &&
          diagnostics.rejection==Rune::TargetRingRejectReason::NoSupportedContour,
          "missing contour rejection did not explain its cause");
}
void reject(const cv::Mat& bgr,Rune::Keypoints k) {
  const auto raw=k;
  require(Rune::refineTargetRing(bgr,k,Rune::Mode::Small,ratio)==Rune::TargetRingResult::Rejected,
          "unverified target ring was accepted");
  require(std::none_of(k.point_valid.begin(),k.point_valid.begin()+4,[](bool v){return v;}) &&
          k.point_valid[4]==raw.point_valid[4] && target(k)==target(raw) && k.center_r==raw.center_r,
          "rejection fabricated coordinates or changed R validity");
}
void checkRejections() {
  reject(image(false),points(.45F));
  reject(cv::Mat(600,800,CV_8UC3,cv::Scalar::all(0)),points());
  auto bgr=image();cv::rectangle(bgr,{395,280},{475,425},cv::Scalar::all(8),cv::FILLED);
  reject(bgr,points());
  bgr=cv::Mat(600,800,CV_8UC3,cv::Scalar::all(8));
  cv::circle(bgr,center,60,{255,30,10},cv::FILLED);reject(bgr,points());
  reject(image(true,{230,230,230}),points());
  auto weak=points(.45F);weak.point_valid[0]=false;
  const auto weak_top=weak.top;
  require(Rune::refineTargetRing(image(),weak,Rune::Mode::Small,ratio)==Rune::TargetRingResult::Refined &&
          !weak.point_valid[0] && weak.top==weak_top && weak.point_valid[1] &&
          std::abs(cv::norm(weak.left-center)-60)<3,
          "partial target was discarded or a weak point was promoted");
  weak=points();weak.point_valid[0]=weak.point_valid[2]=false;reject(image(),weak);
  weak=points();weak.point_valid[4]=false;reject(image(),weak);
  auto wrong=points();wrong.center_r={400,210};reject(image(),wrong);
  auto edge=points();
  for (auto* p : {&edge.top,&edge.left,&edge.right,&edge.bottom,&edge.center_r}) p->x-=350;
  reject(image(),edge);
  auto activated=points();activated.class_id=1;
  require(Rune::refineTargetRing(image(false),activated,Rune::Mode::Big,ratio)==Rune::TargetRingResult::Skipped &&
          activated.point_valid==points().point_valid,"arm-only big blade was treated as a missing ring");
  require(Rune::refineTargetRing(image(true,{10,30,255}),activated,Rune::Mode::Small,ratio)==Rune::TargetRingResult::Refined,
          "red/small-activated ring was rejected");
}
void checkConfig() {
  const auto geometry=YAML::Load("radius_m: .7\narmor_half_width_m: .15\narmor_half_height_m: .15");
  const auto config=Rune::KeypointDetectorConfig::fromYaml(YAML::Load("lock_target_ring: true"),{},geometry);
  require(config.lock_target_ring && std::abs(config.target_ring_radius_ratio-.15/.7)<1e-9,
          "ring dimensions were not derived from measured geometry");
  for (const char* yaml : {"{}","radius_m: 0\narmor_half_width_m: .15\narmor_half_height_m: .15",
       "radius_m: .7\narmor_half_width_m: .nan\narmor_half_height_m: .15",
       "radius_m: .7\narmor_half_width_m: .2\narmor_half_height_m: .15"}) {
    bool threw=false;
    try { (void)Rune::KeypointDetectorConfig::fromYaml(
        YAML::Load("lock_target_ring: true"),{},YAML::Load(yaml)); }
    catch (const std::invalid_argument&) { threw=true; }
    require(threw,"invalid measured ring geometry was accepted");
  }
}

cv::Mat photometricImage(cv::Point2f c,cv::Matx22f plane,float sigma,float exposure,bool reflection) {
  cv::Mat linear(600,800,CV_32FC3,cv::Scalar::all(8));
  for (int radius : {42,60}) for (int i=0;i<72;++i) {
    const float angle=float(2*CV_PI*i/72);
    const auto q=plane*cv::Vec2f(radius*std::cos(angle),radius*std::sin(angle));
    const auto p=c+cv::Point2f(q[0],q[1]);
    const int extent=cvCeil(4*sigma);
    for (int y=cvFloor(p.y)-extent;y<=cvCeil(p.y)+extent;++y)
      for (int x=cvFloor(p.x)-extent;x<=cvCeil(p.x)+extent;++x) {
        if (x<0 || x>=linear.cols || y<0 || y>=linear.rows) continue;
        const float d=(x-p.x)*(x-p.x)+(y-p.y)*(y-p.y);
        const float halo=exposure*std::exp(-d/(2*sigma*sigma));
        const float white=.8F*exposure*std::exp(-d/(.72F*sigma*sigma));
        linear.at<cv::Vec3f>(y,x)+=cv::Vec3f(halo+white,.10F*halo+white,.06F*halo+white);
      }
  }
  cv::Mat bgr;linear.convertTo(bgr,CV_8UC3);
  if (reflection) {
    const auto q=plane*cv::Vec2f(60*std::cos(.4F),60*std::sin(.4F));
    cv::circle(bgr,c+cv::Point2f(q[0]+3,q[1]+2),4,{255,255,255},cv::FILLED,cv::LINE_AA);
  }
  return bgr;
}

void checkSubpixelAccuracy() {
  double coarse_error=0,fine_error=0;
  int accepted=0;
  for (int i=0;i<12;++i) {
    const cv::Point2f c(400.15F+.23F*i,350.35F-.17F*i);
    const cv::Matx22f plane=i%2 ? cv::Matx22f(.72F,.15F,0,1) : cv::Matx22f::eye();
    const float sigma=.9F+.4F*(i%3),exposure=i%2 ? 600.0F : 150.0F;
    const auto bgr=photometricImage(c,plane,sigma,exposure,i%4==0);
    auto truth=points();
    for (auto* p : {&truth.top,&truth.left,&truth.right,&truth.bottom,&truth.center_r}) {
      const auto q=plane*cv::Vec2f(p->x-center.x,p->y-center.y);*p=c+cv::Point2f(q[0],q[1]);
    }
    auto coarse=truth,fine=truth;
    require(Rune::refineTargetRing(bgr,coarse,Rune::Mode::Small,ratio,false)==Rune::TargetRingResult::Refined &&
            Rune::refineTargetRing(bgr,fine,Rune::Mode::Small,ratio,true)==Rune::TargetRingResult::Refined,
            "analytic LED fixture lost its verified ring");
    accepted+=fine.target_ring_subpixel;
    const auto expected=target(truth),a=target(coarse),b=target(fine);
    for (int j=0;j<4;++j) {
      coarse_error+=cv::norm(a[j]-expected[j]);fine_error+=cv::norm(b[j]-expected[j]);
      if (cv::norm(b[j]-expected[j])>=.9)
        std::cerr<<"fixture="<<i<<" point="<<j<<" subpixel="<<fine.target_ring_subpixel
                 <<" error="<<cv::norm(b[j]-expected[j])<<'\n';
      require(cv::norm(b[j]-expected[j])<.9,"subpixel ring missed analytic LED centerline");
    }
    require(fine.center_r==truth.center_r && fine.point_valid==truth.point_valid &&
            fine.point_confidences==truth.point_confidences,"subpixel fitting changed R or scores");
  }
  std::cout<<"LED centerline fixture: coarse_mae="<<coarse_error/48
           <<" subpixel_mae="<<fine_error/48<<" refined="<<accepted<<"/12\n";
  require(accepted==12 && fine_error<.25*coarse_error,
          "subpixel stage did not reduce halo/overexposure bias");
}

void checkSubpixelFallbackAndClutter() {
  // Color still verifies the circle, but a bright neutral background provides
  // too little luminance contrast for an independent subpixel ridge.
  cv::Mat bgr(600,800,CV_8UC3,cv::Scalar::all(250));
  cv::circle(bgr,center,60,{255,30,10},2,cv::LINE_AA);
  auto coarse=points(),fine=points();
  require(Rune::refineTargetRing(bgr,coarse,Rune::Mode::Small,ratio,false)==Rune::TargetRingResult::Refined &&
          Rune::refineTargetRing(bgr,fine,Rune::Mode::Small,ratio,true)==Rune::TargetRingResult::Refined &&
          !fine.target_ring_subpixel && target(fine)==target(coarse),
          "failed photometric refinement changed the already verified coarse measurement");
  // A nearby inner circle is brighter than the intended outer rim.
  bgr=cv::Mat(600,800,CV_8UC3,cv::Scalar::all(8));
  cv::circle(bgr,center,60,{155,5,3},2,cv::LINE_AA);
  cv::circle(bgr,center,54,{255,30,10},2,cv::LINE_AA);
  fine=points(.45F);
  require(Rune::refineTargetRing(bgr,fine,Rune::Mode::Small,ratio,true)==Rune::TargetRingResult::Refined &&
          fine.target_ring_subpixel,"a brighter nested circle prevented outer-ridge fitting");
  for (const auto& p : target(fine))
    require(std::abs(cv::norm(p-center)-60)<.7,"subpixel refinement switched to the brighter inner circle");
  // A connected colored spur is compatible with coarse ring verification but
  // its extreme pixels must not control the final ellipse.
  bgr=image();
  cv::line(bgr,center+cv::Point2f(58,0),center+cv::Point2f(76,0),{255,30,10},3,cv::LINE_AA);
  fine=points();
  require(Rune::refineTargetRing(bgr,fine,Rune::Mode::Small,ratio,true)==Rune::TargetRingResult::Refined &&
          fine.target_ring_subpixel,"a local colored spur prevented robust rim fitting");
  for (const auto& p : target(fine))
    require(std::abs(cv::norm(p-center)-60)<.8,"a colored spur pulled the refined rim off its centerline");
}
} // namespace
int main() {
  try { checkSwitching();checkMotion();checkDisconnectedLeds();checkRejections();checkConfig();checkSubpixelAccuracy();
    checkSubpixelFallbackAndClutter();
    std::cout<<"rune_target_ring_smoke passed\n";return 0; }
  catch (const std::exception& e) { std::cerr<<"rune_target_ring_smoke: "<<e.what()<<'\n';return 1; }
}
