#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l6_telemetry/logger.hpp"
#include "vision_tools/recording/recorder.hpp"

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

using Clock = std::chrono::steady_clock;
constexpr const char* kPreviewWindow = "record_capture";

const char* kCommandLineKeys =
  "{help h usage ? |                           | Show command line help}"
  "{@camera-config| config/camera_config.yaml | Camera YAML path}"
  "{serial-config s| config/serial_config.yaml | Serial YAML path}"
  "{output-dir o   | records                   | Output directory}"
  "{fps f          | 90.0                      | Maximum recorded FPS}"
  "{duration d     | 0                         | Seconds to record; 0 means until Ctrl+C}"
  "{preview        | true                      | Show live preview; Q/Esc stops recording}"
  "{pose-wait-ms   | 20                        | Maximum wait for a pose after each frame}";

volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int)
{
  stop_requested = 1;
}

class PreviewWindow {
public:
  explicit PreviewWindow(bool enabled) : enabled_(enabled)
  {
    if (enabled_) {
      cv::namedWindow(kPreviewWindow, cv::WINDOW_NORMAL);
    }
  }

  ~PreviewWindow()
  {
    if (enabled_) {
      try {
        cv::destroyWindow(kPreviewWindow);
      } catch (const cv::Exception&) {
        // The user or the window manager may have already closed the window.
      }
    }
  }

  void show(const cv::Mat& frame, std::uint64_t submitted,
            std::uint64_t missing_pose, bool has_pose)
  {
    if (!enabled_) {
      return;
    }
    // Resize into a separate image so overlays never enter the recorded video.
    cv::Mat preview;
    cv::resize(frame, preview, {}, 0.5, 0.5, cv::INTER_AREA);
    const cv::Scalar color = has_pose ? cv::Scalar(0, 255, 0)
                                     : cv::Scalar(0, 165, 255);
    cv::putText(preview,
                "Queued: " + std::to_string(submitted)
                  + "  Missing pose: " + std::to_string(missing_pose),
                {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv::LINE_AA);
    cv::putText(preview,
                has_pose ? "Q / Esc: stop" : "No aligned pose - frame skipped",
                {15, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                cv::LINE_AA);
    cv::imshow(kPreviewWindow, preview);
  }

  bool shouldStop() const
  {
    if (!enabled_) {
      return false;
    }
    const int key = cv::waitKey(1);
    return key == 27 || key == 'q' || key == 'Q'
           || cv::getWindowProperty(kPreviewWindow, cv::WND_PROP_VISIBLE) < 1;
  }

private:
  bool enabled_;
};

std::optional<Eigen::Quaterniond> waitForPose(
  const L1Sensor::SerialWorker& serial,
  Clock::time_point frame_timestamp,
  std::chrono::milliseconds timeout)
{
  const auto deadline = Clock::now() + timeout;
  while (true) {
    if (serial.poseReady(frame_timestamp)) {
      return serial.gimbalPoseAt(frame_timestamp);
    }
    if (stop_requested != 0 || Clock::now() >= deadline) {
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
}

int runCapture(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv, kCommandLineKeys);
  parser.about("Record camera frames with time-aligned gimbal quaternions");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  const std::string camera_config = parser.get<std::string>(0);
  const std::string serial_config_path =
    parser.get<std::string>("serial-config");
  const std::filesystem::path output_dir =
    parser.get<std::string>("output-dir");
  const double fps = parser.get<double>("fps");
  const int duration_seconds = parser.get<int>("duration");
  const int pose_wait_ms = parser.get<int>("pose-wait-ms");
  const bool preview_enabled = parser.get<bool>("preview");
  if (!parser.check()) {
    parser.printErrors();
    return 2;
  }
  if (camera_config.empty() || serial_config_path.empty()) {
    throw std::invalid_argument("camera and serial config paths must not be empty");
  }
  if (output_dir.empty()) {
    throw std::invalid_argument("output directory must not be empty");
  }
  if (!std::isfinite(fps) || fps <= 0.0) {
    throw std::invalid_argument("fps must be a positive finite number");
  }
  if (duration_seconds < 0 || pose_wait_ms < 0) {
    throw std::invalid_argument("duration and pose-wait-ms must not be negative");
  }

  auto serial_config =
    L1Sensor::loadSerialConfig(serial_config_path);
  if (!serial_config.enable) {
    throw std::runtime_error("serial is disabled in " + serial_config_path);
  }
  // 录制工具只被动接收姿态，绝不向云台发送控制命令。
  serial_config.tx_enable = false;

  L1Sensor::SerialWorker serial{serial_config};
  if (!serial.start()) {
    throw std::runtime_error("failed to start serial worker");
  }

  L6Telemetry::logInfo(
    "starting camera capture; only frames with aligned gimbal poses will be saved",
    "serial", serial_config.device);

  // Start the camera and preview even before the serial stream supplies a pose.
  // Per-frame pose checks below still gate all writes to the recorder.
  L1Sensor::Camera camera{camera_config};
  vision_tools::Recorder recorder{vision_tools::RecorderConfig{
    .enabled = true,
    .fps = fps,
    .output_dir = output_dir}};
  PreviewWindow preview{preview_enabled};

  const auto pose_wait = std::chrono::milliseconds{pose_wait_ms};
  const auto deadline = duration_seconds == 0
                          ? Clock::time_point::max()
                          : Clock::now()
                              + std::chrono::seconds{duration_seconds};
  std::uint64_t captured_count = 0;
  std::uint64_t submitted_count = 0;
  std::uint64_t missing_pose_count = 0;

  L6Telemetry::logInfo(
    "record capture started",
    "camera_config", camera_config,
    "serial_config", serial_config_path,
    "output", output_dir.string(),
    "fps", fps,
    "pose_wait_ms", pose_wait_ms);
  std::cout << "Capture started (saving requires aligned gimbal poses); press Ctrl+C"
            << (preview_enabled ? " or Q/Esc in the preview window" : "")
            << " to stop.\n";

  cv::Mat frame;
  Clock::time_point timestamp;
  while (stop_requested == 0 && Clock::now() < deadline) {
    if (!camera.read(frame, timestamp)) {
      if (preview.shouldStop()) {
        break;
      }
      continue;
    }
    ++captured_count;

    const auto pose = waitForPose(serial, timestamp, pose_wait);
    if (!pose) {
      ++missing_pose_count;
    } else if (recorder.record(frame, *pose, timestamp)) {
      ++submitted_count;
    }
    preview.show(frame, submitted_count, missing_pose_count, pose.has_value());
    if (preview.shouldStop()) {
      break;
    }
  }

  camera.stop();
  recorder.stop();
  serial.stop();
  L6Telemetry::logInfo(
    "record capture stopped",
    "captured", captured_count,
    "submitted", submitted_count,
    "missing_pose", missing_pose_count,
    "serial_received", serial.receivedStateCount(),
    "serial_skipped_bytes", serial.skippedByteCount(),
    "serial_dropped", serial.droppedPacketCount());
  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  L6Telemetry::initLogger();
  std::signal(SIGINT, requestStop);
  std::signal(SIGTERM, requestStop);

  try {
    const int result = runCapture(argc, argv);
    L6Telemetry::flushLogger();
    return result;
  } catch (const std::exception& error) {
    L6Telemetry::logError("record capture failed", error.what());
    L6Telemetry::flushLogger();
    return 1;
  }
}
