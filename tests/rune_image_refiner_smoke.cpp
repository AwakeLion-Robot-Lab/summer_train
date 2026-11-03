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
  try { check(); checkWithoutSerial(); std::cout<<"rune_image_refiner_smoke passed\n"; return 0; }
  catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
