#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

namespace L1Sensor {

struct CameraStats {
  std::uint64_t captured_frames{};
  std::uint64_t dropped_frames{};
};

class CameraBase {
public:
  virtual ~CameraBase() = default;
  virtual bool read(
    cv::Mat& img,
    std::chrono::steady_clock::time_point& timestamp,
    std::chrono::milliseconds timeout) = 0;
  virtual void stop() = 0;
  virtual CameraStats stats() const { return {}; }
  // 在线调参：曝光（ms）与增益（dB）。返回 false 表示该驱动不支持热更。
  // Talos/回放等后端没有硬件可写，默认 false。
  virtual bool setExposureGain(double /*exposure_ms*/, double /*gain*/)
  {
    return false;
  }
};

class Camera {
public:
  explicit Camera(const std::string& config_path);
  bool read(
    cv::Mat& img,
    std::chrono::steady_clock::time_point& timestamp,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{50});
  void stop();
  CameraStats stats() const;

  // 标定参数未配置时返回空；L2 应以图像曝光时刻对应的内参和畸变参数进行 PnP。
  const std::optional<CameraCalibration>& calibration() const noexcept;

  // 透传到驱动。Talos/文件回放返回 false。
  bool setExposureGain(double exposure_ms, double gain);

private:
  std::unique_ptr<CameraBase> camera_;
  std::optional<CameraCalibration> calibration_;
};

}  // namespace L1Sensor
