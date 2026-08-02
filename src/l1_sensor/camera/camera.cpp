#include "l1_sensor/camera/camera.hpp"

#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
#include "l6_telemetry/logger.hpp"
#include "mindvision/mindvision.hpp"
#include "yaml.hpp"

namespace L1Sensor {

Camera::Camera(const std::string &config_path) {
  const auto config = tools::load(config_path);
  const auto camera_name = tools::read<std::string>(config, "camera_name");
  const auto exposure_ms = tools::read<double>(config, "exposure_ms");
  const auto vid_pid = tools::read<std::string>(config, "vid_pid");

  if (const auto calibration = config["calibration"]) {
    calibration_ =
        loadCameraCalibration(calibration, config_path + ": calibration");
    L6Telemetry::logInfo("camera calibration loaded", config_path, "image",
                         calibration_->image_size.width,
                         calibration_->image_size.height,
                         "distortion_coefficients",
                         calibration_->distortion_coefficients.total());
  }

  if (camera_name == "hikrobot") {
    const auto gain = tools::read<double>(config, "gain");
    camera_ = std::make_unique<io::HikRobot>(exposure_ms, gain, vid_pid);
    return;
  }

  if (camera_name == "mindvision") {
    const auto gamma = tools::read<double>(config, "gamma");
    camera_ = std::make_unique<io::MindVision>(exposure_ms, gamma, vid_pid);
    return;
  }

  throw std::runtime_error("Unsupported camera_name: " + camera_name);
}

bool Camera::read(cv::Mat &img,
                  std::chrono::steady_clock::time_point &timestamp,
                  std::chrono::milliseconds timeout) {
  if (!camera_) {
    throw std::runtime_error("Camera backend is not initialized.");
  }
  return camera_->read(img, timestamp, timeout);
}

void Camera::stop() {
  if (camera_) {
    camera_->stop();
  }
}

const std::optional<CameraCalibration> &Camera::calibration() const noexcept {
  return calibration_;
}

} // namespace L1Sensor
