#include "l1_sensor/camera/camera.hpp"
#include "l6_telemetry/logger.hpp"
#include "tools/camera_calibration/timed_image_saver.hpp"

#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr const char* kWindowName = "calibration_capture";

const char* kCommandLineKeys =
  "{help h usage ? |                           | Show command line help}"
  "{@config-path  | config/carmera_config.yaml | Camera YAML path}"
  "{output-dir o  | calibration_images         | Image output directory}"
  "{fps f         | 1.0                        | Saved images per second}";

bool shouldExit(int key) noexcept
{
  return key == 27 || key == 'q' || key == 'Q';
}

void drawStatus(
  cv::Mat& preview,
  const tools::TimedImageSaver& saver)
{
  const cv::Scalar text_color{0, 255, 0};
  cv::putText(
    preview,
    "Saved: " + std::to_string(saver.savedCount()),
    {20, 35},
    cv::FONT_HERSHEY_SIMPLEX,
    0.8,
    text_color,
    2,
    cv::LINE_AA);
  cv::putText(
    preview,
    "Output: " + saver.sessionDirectory().string(),
    {20, 70},
    cv::FONT_HERSHEY_SIMPLEX,
    0.6,
    text_color,
    2,
    cv::LINE_AA);
}

int runCapture(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv, kCommandLineKeys);
  parser.about("Automatically save raw camera frames for calibration");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  const std::string config_path = parser.get<std::string>(0);
  const std::string output_dir = parser.get<std::string>("output-dir");
  const double fps = parser.get<double>("fps");
  if (!parser.check()) {
    parser.printErrors();
    return 2;
  }
  if (config_path.empty()) {
    throw std::invalid_argument("camera config path is empty");
  }

  tools::TimedImageSaver saver{
    tools::TimedImageSaverConfig{
      .fps = fps,
      .output_dir = std::filesystem::path{output_dir}}};
  L1Sensor::Camera camera{config_path};

  L6Telemetry::logInfo(
    "camera capture started",
    "config", config_path,
    "fps", fps,
    "output", saver.sessionDirectory().string());

  cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;

  while (true) {
    if (!camera.read(frame, timestamp)) {
      if (shouldExit(cv::waitKey(1))) {
        break;
      }
      continue;
    }

    saver.save(frame, timestamp);

    cv::Mat preview;
    cv::resize(frame, preview, {}, 0.5, 0.5, cv::INTER_AREA);
    drawStatus(preview, saver);
    cv::imshow(kWindowName, preview);
    if (shouldExit(cv::waitKey(1))) {
      break;
    }
  }

  camera.stop();
  cv::destroyWindow(kWindowName);
  L6Telemetry::logInfo(
    "camera capture stopped",
    "saved", saver.savedCount(),
    "output", saver.sessionDirectory().string());
  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  L6Telemetry::initLogger();
  try {
    const int result = runCapture(argc, argv);
    L6Telemetry::flushLogger();
    return result;
  } catch (const std::exception& error) {
    cv::destroyAllWindows();
    L6Telemetry::logError("camera capture failed", error.what());
    L6Telemetry::flushLogger();
    return 1;
  }
}
