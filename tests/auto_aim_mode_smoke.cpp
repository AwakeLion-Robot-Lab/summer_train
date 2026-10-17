#include "runtime/auto_aim_config.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct ConfigFile {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("auto-aim-mode-" + std::to_string(getpid()) + ".yaml");
  ~ConfigFile() { std::filesystem::remove(path); }
  runtime::AutoAimConfig load(const std::string& yaml) {
    { std::ofstream out(path); out << yaml; }
    return runtime::loadAutoAimConfig(path.string());
  }
};
struct Detector final : Rune::IKeypointDetector {
  Rune::Mode seen = Rune::Mode::Small;
  std::vector<Rune::Keypoints> points;
  std::vector<Rune::Keypoints> detect(const cv::Mat&) override { return points; }
  std::vector<Rune::Keypoints> detect(const cv::Mat&, Rune::Mode mode) override {
    seen = mode;
    return points;
  }
};

Rune::Frame scene() {
  Rune::Frame frame;
  frame.bgr = cv::Mat(600, 800, CV_8UC3, cv::Scalar(0));
  frame.camera_matrix = (cv::Mat_<double>(3, 3) << 1000,0,400, 0,1000,300, 0,0,1);
  frame.distortion = cv::Mat::zeros(1, 5, CV_64F);
  Eigen::Isometry3d extrinsics = Eigen::Isometry3d::Identity();
  extrinsics.linear() << 0,0,1, -1,0,0, 0,-1,0;
  frame.T_barrel_camera = extrinsics;
  frame.q_world_barrel = Eigen::Quaterniond::Identity();
  // One front-facing inactive blade, 7 m ahead. Dimensions are test data only.
  const auto project = [&](const std::vector<cv::Point3d>& points) {
    std::vector<cv::Point2f> pixels;
    std::vector<cv::Point3f> model;
    for (auto p : points) model.emplace_back(p.x, p.y, p.z);
    cv::projectPoints(model, cv::Vec3d(0,0,0), cv::Vec3d(0,0,7),
                      frame.camera_matrix, frame.distortion, pixels);
    return pixels;
  };
  const auto anchors = project({{0,.8,0},{-.12,.7,0},{.12,.7,0},{0,.6,0},{0,0,0}});
  frame.detections.push_back({anchors[0],anchors[1],anchors[2],anchors[3],anchors[4],0,1});
  for (const auto& bounds : {std::array<double,3>{.12,.6,.8},
                            std::array<double,3>{.035,.15,.5},
                            std::array<double,3>{.025,-.025,.025}}) {
    const auto pixels = project({{-bounds[0],bounds[1],0},{bounds[0],bounds[1],0},
                                 {bounds[0],bounds[2],0},{-bounds[0],bounds[2],0}});
    std::vector<cv::Point> contour;
    for (auto p : pixels) contour.emplace_back(cvRound(p.x), cvRound(p.y));
    cv::fillPoly(frame.bgr, std::vector<std::vector<cv::Point>>{contour}, {0,0,255});
  }
  frame.captured_at = std::chrono::steady_clock::now();
  return frame;
}
}  // namespace

int main() {
  check(runtime::loadAutoAimConfig("config/auto_aim.yaml").mode == runtime::AimMode::Armor,
        "shipped configuration mode changed");
  ConfigFile file;
  for (const auto& [number, expected] : {
         std::pair{1, L1Sensor::WorkMode::AutoAim},
         std::pair{2, L1Sensor::WorkMode::SmallBuff},
         std::pair{3, L1Sensor::WorkMode::BigBuff}}) {
    const auto config = file.load("mode: " + std::to_string(number));
    check(runtime::workMode(config.mode) == expected, "mode mapping failed");
    check(!config.rune.fire.shoot_enable, "rune firing must default to disabled");
  }
  check(file.load("{}").mode == runtime::AimMode::Armor, "missing mode default failed");
  for (const auto value : {"0", "4", "-1", "2.5", "small_buff", "null"}) {
    bool rejected = false;
    try { (void)file.load(std::string("mode: ") + value); }
    catch (const std::exception&) { rejected = true; }
    check(rejected, "invalid mode silently accepted");
  }
  const auto config = file.load(
      "mode: 3\nrune:\n  detector_config: rune.yaml\n  mpc_enable: false\n"
      "  geometry: {radius_m: 0.7, armor_half_width_m: 0.12, armor_half_height_m: 0.1}\n"
      "  fire: {shoot_enable: true, min_shot_interval_s: 0.6}\n");
  check(config.rune.detector_config == file.path.parent_path() / "rune.yaml", "relative detector path failed");
  check(config.rune.mpc_enable == false && config.rune.fire.shoot_enable &&
        config.rune.geometry.radius_m == .7 && config.rune.fire.min_shot_interval_s == .6,
        "rune configuration failed");

  runtime::RuneRuntimeConfig settings;
  settings.geometry = {.7,.12,.1};
  settings.geometry.use_pca_anchors = false;
  settings.geometry.refinement_iterations = 0;
  settings.mpc_enable = false;
  L4Planning::PlannerTuning tuning;
  tuning.default_bullet_speed = 27;
  auto detector = std::make_unique<Detector>();
  auto* observed = detector.get();
  runtime::RuneRuntime pipeline(settings, tuning, std::move(detector));
  for (const auto mode : {Rune::Mode::Small, Rune::Mode::Big}) {
    auto frame = scene();
    frame.mode = mode;
    observed->points = frame.detections;
    frame.detections.clear();
    pipeline.detect(frame);
    check(observed->seen == mode && frame.detections.size() == 1,
          "detector did not receive selected rune mode");
    L1Sensor::RobotState state;
    state.enemy_color = L1Sensor::EnemyColor::Red;
    state.timestamp = frame.captured_at;
    const auto now = frame.captured_at + std::chrono::milliseconds(5);
    const auto command = pipeline.process(frame, state, Eigen::Quaterniond::Identity(), now);
    check(command && !command->shoot && pipeline.lastCommand().found &&
          pipeline.lastCommand().center_hold, "runtime did not produce warmup center hold");
    check(command->pitch < 0, "upward ballistic pitch not converted to serial convention");
    state.bullet_speed = 27;
    const auto explicit_speed = pipeline.process(frame, state, Eigen::Quaterniond::Identity(), now);
    check(explicit_speed && std::abs(command->pitch - explicit_speed->pitch) < 1e-6,
          "configured bullet speed fallback differs from MCU speed");
    for (int failure = 0; failure < 5; ++failure) {
      auto invalid = frame;
      auto feedback = state;
      if (failure == 0) invalid.q_world_barrel.reset();
      if (failure == 1) invalid.T_barrel_camera.reset();
      if (failure == 2) feedback.enemy_color = L1Sensor::EnemyColor::Unknown;
      if (failure == 3) feedback.timestamp -= std::chrono::seconds(1);
      const std::optional<Eigen::Quaterniond> actual = failure == 4
          ? std::nullopt : std::optional{Eigen::Quaterniond::Identity()};
      const auto hold = pipeline.process(invalid, feedback, actual, now);
      check(hold && !hold->shoot && !pipeline.lastCommand().found &&
            hold->yaw_velocity == 0 && hold->pitch_velocity == 0,
            "invalid sensor data did not hold safely");
    }
  }
  std::cout << "auto aim mode/runtime checks passed\n";
}
