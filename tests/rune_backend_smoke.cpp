#include "runtime/auto_aim_config.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct TempConfig {
  std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("rune-backend-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  TempConfig() { std::filesystem::create_directory(directory); }
  ~TempConfig() { std::error_code error; std::filesystem::remove_all(directory, error); }
  std::string write(const std::string& yaml) {
    const auto path = directory / "auto_aim.yaml";
    std::ofstream(path) << yaml;
    return path.string();
  }
};
class ModeProbe final : public Rune::IKeypointDetector {
public:
  std::vector<Rune::Keypoints> detect(const cv::Mat&) override {
    throw std::runtime_error("runtime dropped mode");
  }
  std::vector<Rune::Keypoints> detect(const cv::Mat&, Rune::Mode mode) override {
    last_mode = mode;
    called = true;
    return {};
  }
  std::vector<Rune::Keypoints> detect(const cv::Mat&, Rune::Mode mode, Rune::Color color) override {
    last_color = color;
    return detect(cv::Mat{}, mode);
  }
  bool called{false};
  Rune::Mode last_mode{Rune::Mode::Small};
  Rune::Color last_color{Rune::Color::Blue};
};
}

int main() {
  try {
    TempConfig file;
    for (int mode : {1, 2, 3}) {
      for (bool enabled : {false, true}) {
        const auto config = runtime::loadAutoAimConfig(file.write(
            "mode: " + std::to_string(mode) + "\nfire:\n  shoot_enable: " +
            (enabled ? "true" : "false") +
            "\nrune:\n  fire:\n    shoot_enable: " + (enabled ? "false" : "true") +
            "\n    aim_settle_s: 0.07\n    idle_duration_s: 0.6\n"));
        require(config.fire.shoot_enable == enabled &&
                config.rune.fire.shoot_enable == enabled &&
                config.rune.planning.shoot_enable == enabled,
                "armor and rune did not use the shared top-level firing switch");
        require(config.rune.fire.aim_settle_s == 0.07 &&
                config.rune.fire.idle_duration_s == 0.6,
                "shared firing switch changed rune timing settings");
      }
    }
    const auto default_fire = runtime::loadAutoAimConfig(file.write(
        "rune:\n  fire:\n    shoot_enable: true\n"));
    require(!default_fire.fire.shoot_enable && !default_fire.rune.fire.shoot_enable,
            "legacy rune firing switch bypassed the default shared switch");
    const auto global_only = runtime::loadAutoAimConfig(file.write("fire:\n  shoot_enable: true\n"));
    require(global_only.fire.shoot_enable && global_only.rune.fire.shoot_enable,
            "shared firing switch required a rune section");
    const auto explicit_backend = runtime::loadAutoAimConfig(file.write(
        "rune:\n  detector:\n    backend: openvino\n    model_path: rune-model.onnx\n")).rune;
    const auto implicit_backend = runtime::loadAutoAimConfig(file.write(
        "rune:\n  detector:\n    model_path: rune-model.onnx\n")).rune;
    require(explicit_backend.detector.model_path == file.directory / "rune-model.onnx" &&
        implicit_backend.detector.model_path == explicit_backend.detector.model_path,
        "default and explicit backend configurations resolved different models");
    auto probe = std::make_unique<ModeProbe>();
    auto* state = probe.get();
    runtime::RuneRuntime runtime(implicit_backend, {}, std::move(probe));
    Rune::Frame frame;
    frame.bgr = cv::Mat::zeros(200,200,CV_8UC3);
    frame.mode = Rune::Mode::Big;
    frame.color = Rune::Color::Red;
    runtime.detect(frame);
    require(state->called && state->last_mode == Rune::Mode::Big && state->last_color == Rune::Color::Red,
        "runtime did not forward the selected mode and color");
    const auto scut = runtime::loadAutoAimConfig(file.write(
        "rune:\n  detector:\n    backend: scut\n"
        "    model_path: /missing/model.onnx\n    input_scale: -1\n"
        "    scut:\n      blue_threshold: 23\n"
        "  geometry:\n    radius_m: 0.7\n    armor_half_height_m: 0.14\n")).rune;
    require(scut.scut_detector.blue_threshold == 23 &&
            std::abs(scut.scut_detector.radius_ratio - 0.2) < 1e-9,
            "SCUT configuration was not loaded");
    const auto detector = runtime::makeRuneDetector(scut);
    require(dynamic_cast<Rune::ScutContourDetector*>(detector.get()),
            "SCUT factory did not select the contour backend");
    require(detector->detect(cv::Mat::zeros(200,200,CV_8UC3), Rune::Mode::Small,
                            Rune::Color::Red).empty(),
            "blank image produced a detection");
    bool threw = false;
    try { (void)runtime::loadAutoAimConfig(file.write("rune:\n  detector:\n    backend: typo\n")); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "unknown detector backend was silently accepted");
    std::cout << "rune_backend_smoke passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
