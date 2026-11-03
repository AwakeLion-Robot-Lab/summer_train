#include "rune/f1-observation/keypoint_detector.hpp"
#include "rune/debug_overlay.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <yaml-cpp/yaml.h>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

class Backend final : public L2Perception::IInferenceBackend {
public:
  explicit Backend(bool channels_first) : channels_first_(channels_first) {}
  bool fail{false}, empty{false};
  bool require_fallback{false};
  int inference_count{0};
  float model_scale{1}, pad_x{0}, pad_y{0};
  std::vector<std::uint8_t> last_input;
  std::vector<std::tuple<int, int, float>> changes;
  void load(const L2Perception::InferenceModelConfig&) override {}
  bool ready() const noexcept override { return true; }
  const L2Perception::InferenceInputSpec& inputSpec() const override { return spec_; }
  L2Perception::InferenceResult infer(const L2Perception::InferenceInput& input) override {
    ++inference_count;
    if (fail) throw std::runtime_error("synthetic inference failure");
    last_input.assign(input.values().begin(), input.values().end());
    const bool fallback_padding=last_input[(200*400+75)*3]==114;
    std::vector<float> values(18 * 6, 0);
    const auto set = [&](int anchor, int channel, float value) {
      values[channels_first_ ? channel * 6 + anchor : anchor * 18 + channel] = value;
    };
    const auto candidate = [&](int anchor, int cls, float score, float x, float min_kpt = 0.9F) {
      set(anchor, cls, score);
      for (int point = 0; point < 5; ++point) {
        set(anchor, 3 + 3 * point, (x + point) * model_scale + pad_x);
        set(anchor, 4 + 3 * point, (100 + point) * model_scale + pad_y);
        set(anchor, 5 + 3 * point, point == 2 ? min_kpt : 0.9F);
      }
    };
    if (!empty && (!require_fallback || fallback_padding)) {
      candidate(0, 0, 0.92F, 50);
      candidate(1, 0, 0.80F, 51);  // Duplicate of anchor 0.
      candidate(2, 2, 0.95F, 150);
      candidate(3, 1, 0.90F, 250);
      candidate(4, 0, 0.85F, 350, 0.4F);  // Low-confidence R point.
      candidate(5, 0, 0.20F, 300);
    }
    for (const auto& [anchor, channel, value] : changes) set(anchor, channel, value);
    L2Perception::InferenceTensor tensor;
    tensor.name = "output";
    tensor.shape = channels_first_ ? std::vector<std::size_t>{1, 18, 6}
                                   : std::vector<std::size_t>{1, 6, 18};
    tensor.setOwnedData(std::move(values));
    return {{std::move(tensor)}};
  }

private:
  bool channels_first_;
  L2Perception::InferenceInputSpec spec_{"images", {1, 400, 400, 3}};
};

void check(bool channels_first) {
  auto backend = std::make_unique<Backend>(channels_first);
  auto* source = backend.get();
  Rune::OpenVinoKeypointDetector detector(std::move(backend));
  require(!detector.lastDiagnostics(), "startup probe must not appear as a decoded frame");
  const cv::Mat image(400, 400, CV_8UC3, cv::Scalar::all(0));
  const auto small = detector.detect(image, Rune::Mode::Small);
  const auto d = detector.lastDiagnostics().value();
  require(small.size() == 3 && small[0].class_id == 0 && small[1].class_id == 1 &&
          !small[2].point_valid[4] && small[2].point_confidences[4] == 0.4F,
          "small mode changed the accepted detections");
  require(small[1].top.x == 250 && small[0].center_r.x == 52,
          "diagnostics changed keypoint ordering or coordinates");
  require(d.anchors == 6 && d.score_pass == 5 && d.keypoint_rejected == 0 &&
          d.nms_rejected == 1 && d.mode_rejected == 1 && d.detections == 3 &&
          d.partial_detections == 1 && d.best_valid_keypoints == 5,
          "rejections must identify each filter separately");
  require(d.best_class == 2 && std::abs(d.max_class_scores[0] - 0.92F) < 1e-6 &&
          std::abs(d.max_class_scores[1] - 0.90F) < 1e-6 &&
          std::abs(d.max_class_scores[2] - 0.95F) < 1e-6 &&
          std::abs(d.best_min_keypoint_score - 0.9F) < 1e-6,
          "raw class scores or strongest candidate keypoint confidence are wrong");
  const auto big = detector.detect(image, Rune::Mode::Big);
  require(big.size() == 3 && big[0].top.x == 150 && big[1].top.x == 50,
          "big mode filtering changed");
  source->fail = true;
  bool threw = false;
  try { (void)detector.detect(image); } catch (const std::runtime_error&) { threw = true; }
  require(threw && !detector.lastDiagnostics(), "failure retained previous-frame diagnostics");
  source->fail = false;
  source->empty = true;
  require(detector.detect(image).empty(), "empty model output produced detections");
  const auto blank = detector.lastDiagnostics().value();
  require(blank.score_pass == 0 && blank.detections == 0 && blank.best_class == -1 &&
          blank.max_class_scores[2] == 0, "empty frame retained previous-frame scores");
  require(Rune::formatDetectionDiagnostics(d).find("partial=1") != std::string::npos,
          "diagnostic text omitted the keypoint rejection count");
}

void checkScaledInput(bool channels_first) {
  auto backend = std::make_unique<Backend>(channels_first);
  auto* source = backend.get();
  // 800x400 -> 300x150 inside a 400x400 input, with (50,125) padding.
  source->model_scale = 0.375F;
  source->pad_x = 50;
  source->pad_y = 125;
  auto config = Rune::KeypointDetectorConfig::fromYaml(
      YAML::Load("input_scale: 0.75"), {});
  require(config.input_scale == 0.75F, "input_scale was not loaded from YAML");
  Rune::OpenVinoKeypointDetector detector(std::move(backend), config);
  const auto points = detector.detect(cv::Mat(400, 800, CV_8UC3, cv::Scalar(11, 22, 33)));
  require(points.size() == 3 && points[0].top == cv::Point2f(50, 100) &&
          points[0].center_r == cv::Point2f(52, 102) &&
          points[1].bottom == cv::Point2f(254, 104),
          "scaled and padded keypoints must map back to source pixels");
  require(detector.lastDiagnostics()->nms_rejected == 1,
          "NMS distance must remain in source pixels after resizing");
  const auto pixel = [&](int x, int y, int channel) {
    return source->last_input[(y * 400 + x) * 3 + channel];
  };
  require(pixel(49, 125, 0) == 114 && pixel(50, 124, 2) == 114 &&
          pixel(350, 274, 1) == 114 && pixel(349, 275, 0) == 114 &&
          pixel(50, 125, 0) == 11 && pixel(349, 274, 2) == 33,
          "content size, padding or BGR order changed");
}

void checkInvalidScale() {
  for (const char* value : {"0", "-0.1", "1.1", ".nan", ".inf"}) {
    bool threw = false;
    try {
      (void)Rune::KeypointDetectorConfig::fromYaml(
          YAML::Load(std::string("input_scale: ") + value), {});
    } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "non-finite or out-of-range input_scale must be rejected");
  }
  for (const char* value : {"0", "-1", "6"}) {
    bool threw = false;
    try {
      (void)Rune::KeypointDetectorConfig::fromYaml(
          YAML::Load(std::string("min_valid_keypoints: ") + value), {});
    } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "minimum valid keypoint count must be in [1,5]");
  }
}

void setCandidate(Backend& backend, int anchor, float confidence,
                  const std::array<cv::Point2f,5>& points,
                  const std::array<float,5>& scores) {
  backend.changes.emplace_back(anchor,0,confidence);
  for (int k=0;k<5;++k) {
    backend.changes.emplace_back(anchor,3+3*k,points[k].x);
    backend.changes.emplace_back(anchor,4+3*k,points[k].y);
    backend.changes.emplace_back(anchor,5+3*k,scores[k]);
  }
}

void checkMinimumAndNms(bool channels_first) {
  auto backend=std::make_unique<Backend>(channels_first);
  auto* source=backend.get();
  source->empty=true;
  // Model order top,left,R,right,bottom. The two weak points must not move
  // the NMS center or lower the mean confidence used to rank this candidate.
  const std::array<cv::Point2f,5> points{{{50,100},{60,100},{390,390},{70,100},{390,390}}};
  setCandidate(*source,0,.9F,points,{.9F,.9F,.1F,.9F,.1F});
  const std::array<cv::Point2f,5> duplicate{{{58,100},{59,100},{60,100},{61,100},{62,100}}};
  setCandidate(*source,1,.95F,duplicate,{.85F,.85F,.85F,.85F,.85F});
  Rune::OpenVinoKeypointDetector detector(std::move(backend));
  const cv::Mat image(400,400,CV_8UC3,cv::Scalar::all(0));
  const auto detections=detector.detect(image);
  require(detections.size()==1 && detections[0].top.x==50 &&
          detections[0].point_valid==std::array<bool,5>{true,true,true,false,false},
          "three reliable points must win NMS using only reliable centers and scores");
  require(detector.lastDiagnostics()->nms_rejected==1,
          "weak points moved a duplicate outside the NMS gate");
  cv::Mat overlay=image.clone();
  Rune::drawDetectionOverlay(overlay,detections,Rune::Mode::Small);
  require(overlay.at<cv::Vec3b>(390,390)==cv::Vec3b(0,0,0),
          "overlay drew a weak point or an edge to it");
  source->changes.clear();
  setCandidate(*source,0,.9F,points,{.9F,.9F,.1F,.1F,.1F});
  require(detector.detect(image).empty() && detector.lastDiagnostics()->keypoint_rejected==1,
          "two reliable points must be rejected");
  for (const float bad : {-1.0F,std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity()}) {
    source->changes.clear();
    auto invalid=points;
    invalid[4].x=bad; // Even a weak coordinate must remain finite and nonnegative.
    setCandidate(*source,0,.9F,invalid,{.9F,.9F,.1F,.9F,.1F});
    require(detector.detect(image).empty(), "invalid weak-point coordinates must be rejected");
  }
  source->changes.clear();
  setCandidate(*source,0,.9F,points,{.9F,.9F,.1F,.9F,std::numeric_limits<float>::quiet_NaN()});
  require(detector.detect(image).empty(), "non-finite point confidence must be rejected");
  source->changes.clear();
  auto clipped=points;
  clipped[0]={500,500};
  setCandidate(*source,0,.8F,clipped,{.8F,.8F,.1F,.8F,.1F});
  const auto boundary=detector.detect(image);
  require(boundary.size()==1 && boundary[0].top==cv::Point2f(399,399),
          "threshold equality or image-boundary clipping differs from SZU");

  auto strict_backend=std::make_unique<Backend>(channels_first);
  const auto strict_config=Rune::KeypointDetectorConfig::fromYaml(
      YAML::Load("min_valid_keypoints: 5"),{});
  Rune::OpenVinoKeypointDetector strict(std::move(strict_backend),strict_config);
  require(strict.detect(image).size()==2 && strict.lastDiagnostics()->keypoint_rejected==1,
          "configuring five valid keypoints must recover the previous strict policy");
}

void checkFusion(bool channels_first) {
  auto backend = std::make_unique<Backend>(channels_first);
  auto* source = backend.get();
  source->empty = true;
  const auto config = Rune::KeypointDetectorConfig::fromYaml(YAML::Load(
      "fuse_keypoints: true\nfusion_min_quality_ratio: 0.8\n"
      "fusion_max_point_distance_ratio: 0.2"), {});
  Rune::OpenVinoKeypointDetector detector(std::move(backend), config);
  const cv::Mat image(400,400,CV_8UC3,cv::Scalar::all(0));
  // Model order top,left,R,right,bottom. Opposite target points are 80 px apart.
  const std::array<cv::Point2f,5> points{{{100,140},{60,100},{100,20},{140,100},{100,60}}};
  auto shifted = points;
  for (auto& p : shifted) p.x += 4;
  setCandidate(*source,0,.91F,points,{.9F,.9F,.9F,.9F,.9F});
  setCandidate(*source,1,.90F,shifted,{.9F,.9F,.9F,.9F,.9F});
  const auto before = detector.detect(image);
  require(before.size()==1 && detector.lastDiagnostics()->fused_candidates==1 &&
          before[0].top.x>101 && before[0].top.x<103,
          "consistent duplicate points were not fused");
  require(before[0].confidence==.91F && before[0].point_confidences[0]==.9F,
          "fusion must preserve seed acceptance scores");
  source->changes.emplace_back(1,0,.92F);
  const auto after = detector.detect(image);
  require(after.size()==1 && std::abs(after[0].top.x-before[0].top.x)<1,
          "a small score-order change still caused a full candidate coordinate jump");
  // No state is retained: repeating the first frame returns the first result.
  source->changes.emplace_back(1,0,.90F);
  require(detector.detect(image)[0].top==before[0].top,
          "single-frame fusion retained cross-frame history");

  // An adjacent blade is far from the seed's NMS center; a close candidate
  // with permuted semantic anchors must not contaminate the seed either.
  auto neighbor = points;
  for (auto& p : neighbor) p.x+=100;
  auto permuted = points;
  std::swap(permuted[0],permuted[4]);
  source->changes.clear();
  setCandidate(*source,0,.95F,points,{.9F,.9F,.9F,.9F,.9F});
  setCandidate(*source,1,.94F,permuted,{.9F,.9F,.9F,.9F,.9F});
  setCandidate(*source,2,.93F,neighbor,{.9F,.9F,.9F,.9F,.9F});
  const auto separate = detector.detect(image);
  require(separate.size()==2 && separate[0].top==points[0] &&
          separate[1].top==neighbor[0] && detector.lastDiagnostics()->fused_candidates==0,
          "fusion combined different blades or inconsistent semantic anchors");
  source->changes.clear();
  setCandidate(*source,0,.95F,points,{.9F,.9F,.9F,.9F,.9F});
  setCandidate(*source,1,.94F,shifted,{.9F,.9F,.9F,.9F,.9F});
  source->changes.emplace_back(1,0,0);
  source->changes.emplace_back(1,1,.94F);
  const auto different_class = detector.detect(image);
  require(different_class.size()==1 && different_class[0].top==points[0] &&
          detector.lastDiagnostics()->fused_candidates==0,
          "fusion blended inactive and activated model classes");

  // Weak R coordinates cannot enter a reliable R measurement. Nor can a
  // reliable supporting R promote a weak R belonging to the winning seed.
  source->changes.clear();
  auto weak = shifted;
  weak[2] = {380,380};
  setCandidate(*source,0,.95F,points,{.9F,.9F,.9F,.9F,.9F});
  setCandidate(*source,1,.94F,weak,{.9F,.9F,.1F,.9F,.9F});
  const auto reliable = detector.detect(image);
  require(reliable.size()==1 && reliable[0].center_r==points[2] && reliable[0].top.x>100 &&
          detector.lastDiagnostics()->fused_candidates==1,
          "weak supporting R polluted the seed measurement");
  // Keep the weak R nearby so that both candidates share an NMS group.
  source->changes.clear();
  setCandidate(*source,0,.99F,points,{.9F,.9F,.1F,.9F,.9F});
  setCandidate(*source,1,.90F,shifted,{.9F,.9F,.9F,.9F,.9F});
  const auto partial = detector.detect(image);
  require(partial.size()==1 && !partial[0].point_valid[4] &&
          partial[0].center_r==points[2] && partial[0].point_confidences[4]==.1F &&
          partial[0].top.x>100,
          "fusion promoted a weak seed point or failed to fuse its reliable points");

  // The same source pixels must fuse identically after resize and letterbox.
  auto scaled_backend=std::make_unique<Backend>(channels_first);
  scaled_backend->empty=true;
  auto scaled_config=config;
  scaled_config.input_scale=.75F;
  for (int a=0;a<2;++a) {
    auto scaled_points=a==0 ? points : shifted;
    for (auto& p : scaled_points) p=p*.375F+cv::Point2f(50,125);
    setCandidate(*scaled_backend,a,a==0 ? .91F : .90F,scaled_points,{.9F,.9F,.9F,.9F,.9F});
  }
  Rune::OpenVinoKeypointDetector scaled(std::move(scaled_backend),scaled_config);
  const auto scaled_result=scaled.detect(cv::Mat(400,800,CV_8UC3,cv::Scalar::all(0)));
  require(scaled_result.size()==1 && cv::norm(scaled_result[0].top-before[0].top)<1e-4,
          "fusion gates or coordinates changed with input scale");
}

void checkInvalidFusion() {
  for (const char* yaml : {"fusion_min_quality_ratio: -0.1", "fusion_min_quality_ratio: 1",
       "fusion_min_quality_ratio: .nan", "fusion_max_point_distance_ratio: 0",
       "fusion_max_point_distance_ratio: 1.1", "fusion_max_point_distance_ratio: .inf"}) {
    bool threw=false;
    try { (void)Rune::KeypointDetectorConfig::fromYaml(YAML::Load(yaml),{}); }
    catch (const std::invalid_argument&) { threw=true; }
    require(threw,"invalid fusion gate accepted");
  }
}

void checkTrackingOverlay() {
  Rune::Frame frame;
  frame.bgr=cv::Mat::zeros(400,400,CV_8UC3);
  frame.captured_at=Rune::TimePoint{}+std::chrono::seconds(1);
  frame.camera_matrix=(cv::Mat_<double>(3,3)<<200,0,100,0,200,100,0,0,1);
  // Camera optical origin is offset from barrel by 1m in x. This checks that
  // drawing actually applies the extrinsic, rather than treating barrel as camera.
  frame.T_barrel_camera=Eigen::Isometry3d::Identity();
  frame.T_barrel_camera->translation().x()=1;
  Rune::Pose pose;
  pose.captured_at=frame.captured_at;
  pose.filter_state=Rune::FilterState{};
  pose.center_barrel={1,0,2};
  pose.target_barrel={1.3,0,2};
  auto overlay=frame.bgr.clone();
  Rune::drawTrackingOverlay(overlay,frame,pose);
  require(cv::norm(overlay(cv::Rect(95,95,11,11)),cv::NORM_INF)>0 &&
          cv::norm(overlay(cv::Rect(125,95,11,11)),cv::NORM_INF)>0 &&
          overlay.at<cv::Vec3b>(100,200)==cv::Vec3b(0,0,0),
          "filtered projections ignored extrinsics or used the wrong pixels");
  pose.captured_at-=std::chrono::milliseconds(1);
  overlay=frame.bgr.clone();
  Rune::drawTrackingOverlay(overlay,frame,pose);
  require(cv::norm(overlay,cv::NORM_INF)==0,"overlay drew a stale tracking pose");
  pose.captured_at=frame.captured_at;
  pose.center_barrel.z()=pose.target_barrel.z()=-2;
  Rune::drawTrackingOverlay(overlay,frame,pose);
  require(cv::norm(overlay,cv::NORM_INF)==0,"overlay projected points behind camera");
  frame.camera_matrix.release();
  Rune::drawTrackingOverlay(overlay,frame,pose);
  require(cv::norm(overlay,cv::NORM_INF)==0,"overlay drew without calibration");
}

void checkFallback(bool channels_first) {
  auto backend=std::make_unique<Backend>(channels_first);
  auto* source=backend.get();
  source->require_fallback=true;
  // 800x400 -> 200x100, centered in 400x400 at fallback scale 0.5.
  source->model_scale=.25F; source->pad_x=100; source->pad_y=150;
  const auto config=Rune::KeypointDetectorConfig::fromYaml(
      YAML::Load("input_scale: 0.75\nfallback_input_scale: 0.5"),{});
  Rune::OpenVinoKeypointDetector detector(std::move(backend),config);
  const cv::Mat image(400,800,CV_8UC3,cv::Scalar::all(0));
  int calls=source->inference_count;
  const auto recovered=detector.detect(image);
  require(source->inference_count-calls==2 && detector.lastDiagnostics()->inference_passes==2 &&
          recovered.size()==3 && recovered[0].top==cv::Point2f(50,100) &&
          recovered[0].center_r==cv::Point2f(52,102),
          "fallback inference failed or returned coordinates in model pixels");
  source->require_fallback=false;
  source->model_scale=.375F; source->pad_x=50; source->pad_y=125;
  calls=source->inference_count;
  const auto primary=detector.detect(image);
  require(source->inference_count-calls==1 && detector.lastDiagnostics()->inference_passes==1 &&
          primary.size()==3 && primary[0].top==cv::Point2f(50,100),
          "successful primary detections unnecessarily triggered extra inference");
  source->empty=true;
  calls=source->inference_count;
  require(detector.detect(image).empty() && source->inference_count-calls==2,
          "failed fallback synthesized detections or retried without bound");
  for (const char* value : {"-0.1","1.1",".nan",".inf"}) {
    bool threw=false;
    try { (void)Rune::KeypointDetectorConfig::fromYaml(
        YAML::Load(std::string("fallback_input_scale: ")+value),{}); }
    catch (const std::invalid_argument&) { threw=true; }
    require(threw,"invalid fallback scale accepted");
  }
}
}  // namespace

int main() {
  try {
    check(true);
    check(false);
    checkScaledInput(true);
    checkScaledInput(false);
    checkInvalidScale();
    checkMinimumAndNms(true);
    checkMinimumAndNms(false);
    checkFusion(true);
    checkFusion(false);
    checkInvalidFusion();
    checkTrackingOverlay();
    checkFallback(true);
    checkFallback(false);
    std::cout << "rune_detector_smoke passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rune_detector_smoke: " << error.what() << '\n';
    return 1;
  }
}
