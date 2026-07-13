#pragma once

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
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

private:
  std::unique_ptr<CameraBase> camera_;
};

}  // namespace L1Sensor
