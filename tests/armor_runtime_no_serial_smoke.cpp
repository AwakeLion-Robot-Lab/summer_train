// 直接运行生产主循环；链接替身只提供相机帧与确定的网络输出，不依赖 USB 或模型。
#define main autoAimExecutableMain
#include "../src/runtime/auto_aim_runtime.cpp"
#undef main

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

namespace runtime_test {
int inference_calls = 0;
int frames_left = 0;
bool timeout_once = false;
bool fail_capture = false;
int camera_age_ms = 0;
int stale_reports = 0;
runtime::AutoAimRuntime* active_runtime = nullptr;

class Backend final : public L2Perception::IInferenceBackend {
public:
  void load(const L2Perception::InferenceModelConfig&) override {}
  bool ready() const noexcept override { return true; }
  const L2Perception::InferenceInputSpec& inputSpec() const override { return spec_; }
  L2Perception::InferenceResult infer(const L2Perception::InferenceInput&) override {
    ++inference_calls;
    std::vector<float> row{20, 20, 20, 60, 70, 60, 70, 20, 10};
    row.resize(22, -5);
    row[9] = row[16] = 10;
    L2Perception::InferenceTensor tensor;
    tensor.name = "output";
    tensor.shape = {1, 1, 22};
    tensor.setOwnedData(std::move(row));
    return {{std::move(tensor)}};
  }
private:
  L2Perception::InferenceInputSpec spec_{"images", {1, 100, 100, 3}};
};
}  // namespace runtime_test

namespace L1Sensor {
Camera::Camera(const std::string&) {}
bool Camera::read(cv::Mat& image, Clock::time_point& timestamp, std::chrono::milliseconds) {
  if (runtime_test::fail_capture) throw std::runtime_error("injected camera failure");
  if (std::exchange(runtime_test::timeout_once, false)) return false;
  if (runtime_test::frames_left-- > 0) {
    image = cv::Mat(100, 100, CV_8UC3, cv::Scalar::all(0));
    timestamp = Clock::now() - std::chrono::milliseconds(runtime_test::camera_age_ms);
    return true;
  }
  runtime_test::active_runtime->stop();
  return false;
}
void Camera::stop() {}
const std::optional<CameraCalibration>& Camera::calibration() const noexcept {
  return calibration_;
}
}  // namespace L1Sensor

// Capture telemetry in memory, including the rejection branch before planning.
namespace L6Telemetry {
UdpJsonSender::UdpJsonSender(std::string host, uint16_t port)
  : host_(std::move(host)), port_(port) {}
UdpJsonSender::~UdpJsonSender() = default;
const std::string& UdpJsonSender::host() const noexcept { return host_; }
uint16_t UdpJsonSender::port() const noexcept { return port_; }
bool UdpJsonSender::send(const nlohmann::json& payload) noexcept {
  if (payload.contains("result_rejected")) ++runtime_test::stale_reports;
  return true;
}
} // namespace L6Telemetry

namespace runtime {
L2Perception::ArmorDetector makeDetector(const AutoAimConfig& config, bool) {
  return L2Perception::ArmorDetector(std::make_unique<runtime_test::Backend>(),
    {.decoder = config.decoder, .refiner = config.refiner});
}
}  // namespace runtime

int main() {
  namespace fs = std::filesystem;
  const auto project = fs::current_path();
  char temporary[] = "/tmp/armor-runtime-no-serial-XXXXXX";
  const char* created = mkdtemp(temporary);
  if (!created) return 1;
  const fs::path root(created);
  int status = 0;
  try {
    fs::create_directories(root / "config");
    auto aiming = YAML::LoadFile((project / "config/auto_aim.yaml").string());
    aiming["mode"] = 1;
    aiming["debug"]["overlay"] = false;
    aiming["debug"]["plot"] = false;
    // fake 网络沿用默认的 YOLOv5 输出契约。
    aiming["inference"]["decoder"]["layout"] = "yolov5_22";
    std::ofstream(root / "config/auto_aim.yaml") << aiming;
    fs::copy_file(project / "config/planner_config.yaml", root / "config/planner_config.yaml");
    auto serial = YAML::LoadFile((project / "config/serial_config.yaml").string());
    serial["device"] = (root / "missing-serial").string();
    fs::current_path(root);
    L6Telemetry::initLogger();
    for (const bool pipelined : {false, true}) {
      aiming["runtime"]["armor_pipeline"] = pipelined;
      std::ofstream(root / "config/auto_aim.yaml") << aiming;
      for (const bool enabled : {false, true}) {
        serial["enable"] = enabled;
        std::ofstream(root / "config/serial_config.yaml") << serial;
        runtime_test::inference_calls = 0;
        runtime_test::frames_left = 4;
        runtime_test::timeout_once = true;
        runtime::AutoAimRuntime app("unused-camera.yaml");
        runtime_test::active_runtime = &app;
        app.run();
        // 构造检测器探测输出形状时推理一次；四帧图像都必须继续做检测。
        if (runtime_test::inference_calls != 5) {
          throw std::runtime_error(enabled ? "missing serial device blocked armor detection"
                                           : "disabled serial blocked armor detection");
        }
      }
      // Exceptions in either synchronous capture or the producer must reach the
      // runtime cleanup path without escaping a thread or hanging at shutdown.
      runtime_test::fail_capture = true;
      runtime::AutoAimRuntime failing("unused-camera.yaml");
      runtime_test::active_runtime = &failing;
      failing.run();
      runtime_test::fail_capture = false;
    }
    aiming["runtime"]["armor_pipeline"] = true;
    aiming["runtime"]["pipeline_max_result_age_ms"] = 100;
    aiming["debug"]["plot"] = true;
    std::ofstream(root / "config/auto_aim.yaml") << aiming;
    runtime_test::camera_age_ms = 200;
    runtime_test::frames_left = 4;
    runtime_test::inference_calls = 0;
    runtime::AutoAimRuntime stale("unused-camera.yaml");
    runtime_test::active_runtime = &stale;
    stale.run();
    if (runtime_test::stale_reports == 0 || runtime_test::inference_calls != 5)
      throw std::runtime_error("stale pipeline results were not rejected, or detection stalled");
    L6Telemetry::flushLogger();
    std::cout << "armor runtime without serial smoke passed\n";
  } catch (const std::exception& error) {
    std::cerr << "armor runtime without serial smoke failed: " << error.what() << '\n';
    status = 1;
  }
  fs::current_path(project);
  fs::remove_all(root);
  return status;
}
