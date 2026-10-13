// 录制 auto_aim_test 可直接回放的 records/<name>.avi + .txt。
// TXT 每行是图像时刻（相对首帧的秒数）和 world <- IMU 的 w x y z。
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l6_telemetry/logger.hpp"
#include "tools/record/record_writer.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int)
{
  stop_requested = 1;
}

const char* kKeys =
  "{help h usage ? | false | Show help}"
  "{output o | | Output base path, default records/<local time>}"
  "{fps f | 30 | Maximum recorded FPS and AVI playback FPS}"
  "{duration d | 0 | Capture duration in seconds, 0 = until Ctrl-C}"
  "{max-frames | 0 | Stop after this many paired frames, 0 = unlimited}"
  "{preview | true | Show live camera window; false for headless capture}"
  "{camera-config | config/camera_config.yaml | Camera config}"
  "{serial-config | config/serial_config.yaml | Serial config}";

std::string defaultOutput()
{
  const std::time_t now = std::time(nullptr);
  std::ostringstream path;
  path << "records/" << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S");
  return path.str();
}

int run(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, kKeys);
  if (cli.get<bool>("help")) {
    cli.printMessage();
    return 0;
  }
  const double fps = cli.get<double>("fps");
  const double duration = cli.get<double>("duration");
  const int max_frames = cli.get<int>("max-frames");
  const bool preview = cli.get<bool>("preview");
  const std::string camera_path = cli.get<std::string>("camera-config");
  const std::string serial_path = cli.get<std::string>("serial-config");
  const std::string requested_output = cli.get<std::string>("output");
  if (!cli.check()) {
    cli.printErrors();
    return 2;
  }
  if (!std::isfinite(fps) || fps <= 0.0 ||
      !std::isfinite(duration) || duration < 0.0 || max_frames < 0) {
    throw std::invalid_argument("need fps > 0, duration >= 0, max-frames >= 0");
  }

  const std::string output = requested_output.empty() ? defaultOutput() : requested_output;
  tools::RecordWriter writer{output, fps};
  const auto serial_config = L1Sensor::loadSerialConfig(serial_path);
  if (!serial_config.enable) {
    throw std::runtime_error("serial is disabled; recording needs IMU packets");
  }

  L6Telemetry::initLogger();
  L1Sensor::SerialWorker serial{serial_config, true};  // RX only; no control packets
  if (!serial.start()) {
    throw std::runtime_error("cannot start serial receiver");
  }
  L1Sensor::Camera camera{camera_path};
  const Eigen::Quaterniond q_imu_barrel{serial_config.R_imu_barrel};

  std::signal(SIGINT, requestStop);
  std::signal(SIGTERM, requestStop);
  constexpr const char* window_name = "record camera";
  if (preview) {
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  }
  std::cout << "Recording " << output << ".avi + .txt; "
            << (preview ? "q/Esc in preview or Ctrl-C" : "Ctrl-C")
            << " to stop\n";
  std::cout.flush();

  const auto start = Clock::now();
  const auto min_interval = std::chrono::duration<double>{1.0 / fps};
  Clock::time_point last_recorded{};
  std::size_t camera_frames = 0;
  std::size_t missing_pose = 0;
  cv::Mat image;
  Clock::time_point image_time;

  while (!stop_requested &&
         (duration == 0.0 ||
          std::chrono::duration<double>(Clock::now() - start).count() < duration) &&
         (max_frames == 0 || writer.frames() < static_cast<std::size_t>(max_frames))) {
    if (!camera.read(image, image_time) || image.empty()) {
      if (preview) {
        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) {
          stop_requested = 1;
        }
      }
      continue;
    }
    ++camera_frames;
    if (preview) {
      // 显示原始相机帧；即使缺少姿态或因 FPS 限制跳帧，窗口仍实时更新。
      cv::imshow(window_name, image);
      const int key = cv::waitKey(1);
      if (key == 'q' || key == 'Q' || key == 27) {
        stop_requested = 1;
        break;
      }
    }
    if (writer.frames() > 0 && image_time - last_recorded < min_interval) {
      continue;
    }
    if (const auto& calibration = camera.calibration(); calibration &&
        !calibration->matchesImageSize(image.size())) {
      throw std::runtime_error("camera frame resolution differs from calibration");
    }

    // 与 runtime 同一时刻查姿态：先等图像后的下位机包，再在前后包之间插值。
    const auto pose_time = serial_config.poseTime(image_time);
    serial.waitPose(pose_time);
    const auto span = serial.poseSpan();
    if (!span || pose_time < span->first || pose_time > span->second) {
      ++missing_pose;
      continue;
    }
    const auto q_world_barrel = serial.gimbalPoseAt(pose_time);
    if (!q_world_barrel) {
      ++missing_pose;
      continue;
    }
    // gimbalPoseAt 已经右乘 R_imu_barrel；TXT 需保存原始 IMU 约定，
    // auto_aim_test --convention=imu 才能按当前 serial_config 再复合一次。
    const Eigen::Quaterniond q_world_imu =
      (*q_world_barrel * q_imu_barrel.conjugate()).normalized();
    writer.write(image, q_world_imu, image_time);
    last_recorded = image_time;
  }

  camera.stop();
  serial.stop();
  writer.close();
  if (preview) {
    cv::destroyWindow(window_name);
  }
  std::cout << "Saved " << writer.frames() << " paired frames (camera " << camera_frames
            << ", no bracketed pose " << missing_pose << ") to " << output << "\n";
  if (writer.frames() == 0) {
    std::cerr << "No paired frames: check camera, serial packets and pose_offset_ms\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char* argv[])
{
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "record: " << error.what() << '\n';
    return 1;
  }
}
