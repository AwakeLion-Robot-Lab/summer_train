#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

namespace L1Sensor {

class CameraBase {
public:
  virtual ~CameraBase() = default;
  virtual bool read(
    cv::Mat& img,
    std::chrono::steady_clock::time_point& timestamp,
    std::chrono::milliseconds timeout) = 0;
  virtual void stop() = 0;
};

class Camera {
public:
  explicit Camera(const std::string& config_path);
  bool read(
    cv::Mat& img,
    std::chrono::steady_clock::time_point& timestamp,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{50});
  void stop();

  // 标定参数未配置时返回空；L2 应以图像曝光时刻对应的内参和畸变参数进行 PnP。
  const std::optional<CameraCalibration>& calibration() const noexcept;

private:
  std::unique_ptr<CameraBase> camera_;
  std::optional<CameraCalibration> calibration_;
};

}  // namespace L1Sensor
