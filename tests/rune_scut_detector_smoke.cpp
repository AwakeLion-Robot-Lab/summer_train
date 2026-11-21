#include "rune/f1-observation/scut_contour_detector.hpp"
#include "rune/f1-observation/reobservation.hpp"

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
cv::Mat scene(bool inactive, bool arm, bool hub, bool red = false) {
  cv::Mat image(600, 800, CV_8UC3, cv::Scalar::all(0));
  const cv::Scalar color = red ? cv::Scalar(0, 0, 240) : cv::Scalar(240, 0, 0);
  const cv::Point target(400, 160);
  cv::circle(image, target, 45, color, -1);
  if (inactive) {
    for (int i = 0; i < 4; ++i) {
      const double angle = (45 + i * 90) * CV_PI / 180;
      const cv::Point p = target + cv::Point(cvRound(29 * std::cos(angle)), cvRound(29 * std::sin(angle)));
      cv::ellipse(image, p, {13, 5}, 135 + i * 90, 0, 360, cv::Scalar::all(0), -1);
    }
  } else cv::circle(image, target, 31, cv::Scalar::all(0), -1);
  if (hub) cv::putText(image, "R", {386, 414}, cv::FONT_HERSHEY_SIMPLEX, 1.2, color, 5);
  if (arm && inactive) {
    for (int y : {245, 272, 299, 326}) {
      const std::vector<cv::Point> arrow{{385,y+8},{400,y-6},{415,y+8},
          {410,y+13},{400,y+4},{390,y+13}};
      cv::fillPoly(image, std::vector<std::vector<cv::Point>>{arrow}, color);
    }
  } else if (arm) {
    const std::vector<cv::Point> fan{{365,250},{400,280},{435,250},
        {435,265},{405,310},{395,310},{365,265}};
    cv::fillPoly(image, std::vector<std::vector<cv::Point>>{fan}, color);
  }
  return image;
}
void check(bool inactive, bool red, Rune::Mode mode) {
  Rune::ScutContourDetector detector;
  const auto image = scene(inactive, true, true, red);
  const auto color = red ? Rune::Color::Red : Rune::Color::Blue;
  Rune::Frame frame;
  frame.bgr = image; frame.mode = mode; frame.color = color;
  frame.detections = detector.detect(image, mode, color);
  require(frame.detections.size() == 1, "SCUT must match one target, R and arm");
  const auto& k = frame.detections.front();
  require(k.class_id == (inactive ? 0 : 1), "gap/ring activation classification is wrong");
  require(cv::norm(*k.armorCenter() - cv::Point2f(400,160)) < 2, "target center shifted");
  require(k.top.y < k.bottom.y && k.left.x < k.right.x, "local five-point order is wrong");
  require(k.target_ring_status == Rune::TargetRingStatus::Verified && k.image_contours,
          "measured contours were not retained");
  const auto regions = Rune::DetectionStage{}.runAll(frame);
  require(regions.size() == 1, "SCUT coordinates cannot enter the existing ROI stage");
  const auto observation = Rune::ObservationStage{}.run(regions.front(), color, mode);
  require(observation && !observation->armor.empty() && !observation->arm.empty() &&
          !observation->center_r.empty(), "SCUT cannot enter the existing observation stage");
  require(!Rune::ObservationStage{}.run(regions.front(), red ? Rune::Color::Blue : Rune::Color::Red, mode),
          "semantic contours were reused with the wrong color");
  require(detector.lastDiagnostics()->inference_passes == 0 &&
          detector.lastDiagnostics()->arm_matches == 1, "SCUT diagnostics report inference");
  require(detector.detect(image, mode, red ? Rune::Color::Blue : Rune::Color::Red).empty(),
          "wrong-color targets were accepted");
  require(detector.detect(cv::Mat::zeros(600,800,CV_8UC3), mode, color).empty() &&
          detector.lastDiagnostics()->detections == 0, "blank frame retained detections");
}

void checkLampClutter() {
  Rune::ScutContourDetector detector;
  auto image=scene(true,true,true);
  // These nearly white reflections pass the old absolute B-R threshold but
  // must not enlarge or rotate the measured lamp envelope.
  cv::line(image,{400,345},{400,365},{235,215,195},6);
  cv::circle(image,{450,270},7,{235,215,195},-1);
  // A genuine blue speckle outside the narrow inactive arrow lane must also
  // stay out of the envelope even though its color is valid.
  cv::circle(image,{450,300},5,{240,0,0},-1);
  const auto detections=detector.detect(image);
  require(detections.size()==1,"lamp clutter broke an otherwise valid rune match");
  const auto& measured=*detections[0].image_contours;
  require(measured.fan_points.size()==4,"clean arrow chain lost its measured lamp corners");
  for (const auto& p : measured.fan_points)
    require(p.x>=380 && p.x<=420 && p.y>=232 && p.y<=345,
            "neutral reflections or off-axis blobs contaminated the lamp envelope");
  auto reflections=scene(true,false,true);
  for (const auto y : {245,275,305})
    cv::line(reflections,{390,y},{410,y},{235,215,195},5);
  require(detector.detect(reflections).empty(),"neutral reflections were accepted as an arrow chain");
}

void checkPartialActiveLamp() {
  auto image=scene(false,false,true);
  for (const int x : {370,400,430})
    cv::line(image,{x,245},{x,300},{240,0,0},6);
  Rune::ScutContourDetector detector;
  const auto detections=detector.detect(image,Rune::Mode::Big,Rune::Color::Blue);
  require(detections.size()==1 && detections[0].class_id==1,
          "separate active strips were rejected by the complete-fan aspect-ratio filter");
  const auto& measured=*detections[0].image_contours;
  require(measured.fan_points.size()==3 && measured.fan_point_ids==std::vector<int>{0,1,2},
          "partial active lamp did not publish its three measured strip tips");
}
} // namespace

int main() {
  try {
    for (const auto mode : {Rune::Mode::Small, Rune::Mode::Big})
      for (bool inactive : {false, true}) for (bool red : {false, true}) check(inactive, red, mode);
    checkLampClutter();checkPartialActiveLamp();
    Rune::ScutContourDetector detector;
    const auto original = scene(true,true,true);
    const auto source = detector.detect(original).front();
    const cv::Matx23f transform(.82F, .22F, 20.F, -.12F, .72F, 130.F);
    cv::Mat warped;
    cv::warpAffine(original, warped, transform, original.size());
    const auto projected = detector.detect(warped);
    require(projected.size() == 1, "oblique target was lost");
    const auto apply = [&](cv::Point2f p) {
      const auto value = transform * cv::Vec3f(p.x,p.y,1);
      return cv::Point2f(value[0],value[1]);
    };
    require(cv::norm(projected[0].top - apply(source.top)) < 3 &&
            cv::norm(projected[0].left - apply(source.left)) < 3 &&
            cv::norm(projected[0].right - apply(source.right)) < 3,
            "ellipse axes lost the projected radial/tangential anchor directions");
    require(detector.detect(scene(true,false,true)).empty(), "target without arm was accepted");
    const auto missing_r = detector.detect(scene(true,true,false));
    require(missing_r.empty(), "missing R was fabricated");
    bool threw = false;
    try { (void)detector.detect(cv::Mat{}); } catch (const std::invalid_argument&) { threw = true; }
    require(threw && !detector.lastDiagnostics(), "invalid input retained frame diagnostics");
    for (const char* yaml : {"close_kernel: 0", "close_kernel: 4", 
                             "close_kernel: 20", "blue_threshold: 0", "max_features: 10000",
                             "radius_ratio: .nan", "radius_ratio_tolerance: 0",
                             "min_arm_color_ratio: -0.1", "min_arm_color_ratio: .nan",
                             "min_arm_color_ratio: 1"}) {
      threw = false;
      try { (void)Rune::ScutContourConfig::fromYaml(YAML::Load(yaml), {}); }
      catch (const std::invalid_argument&) { threw = true; }
      require(threw, "invalid SCUT configuration was accepted");
    }
    try {
      (void)Rune::ScutContourConfig::fromYaml(YAML::Load("close_kernel: 20"), {});
      require(false, "invalid close_kernel was accepted");
    } catch (const std::invalid_argument& error) {
      const std::string message = error.what();
      require(message.find("rune.detector.scut.close_kernel") != std::string::npos &&
              message.find("[1, 19]") != std::string::npos &&
              message.find("20") != std::string::npos,
              "close_kernel error must identify the key, valid range and supplied value");
    }
    const auto config = Rune::ScutContourConfig::fromYaml({},
        YAML::Load("radius_m: 0.7\narmor_half_height_m: 0.14"));
    require(std::abs(config.radius_ratio - 0.2) < 1e-9, "measured geometry was not used");
    std::cout << "rune_scut_detector_smoke passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
