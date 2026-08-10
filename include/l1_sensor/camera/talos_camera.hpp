#pragma once

#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/talos/talos_config.hpp"
#include "l1_sensor/talos/talos_reader.hpp"

#include <chrono>
#include <memory>
#include <optional>

namespace L1Sensor {

// Daedalus 仿真器 Talos 共享内存相机后端。
// 共享同一个 TalosReader 时，TalosSerial 与 TalosCamera 可保持帧/位姿同步。
class TalosCamera final : public CameraBase {
public:
  // 从相机配置 YAML 自行创建 reader（Camera 包装器路径）。
  explicit TalosCamera(const std::string& config_path);

  // 复用外部共享 reader（talos_auto_aim 工具路径）。
  TalosCamera(
    std::shared_ptr<talos::TalosReader> reader,
    talos::TalosSimConfig config);

  bool read(
    cv::Mat& img,
    std::chrono::steady_clock::time_point& timestamp,
    std::chrono::milliseconds timeout) override;

  void stop() override;

  [[nodiscard]] const std::optional<CameraCalibration>& calibration()
    const noexcept
  {
    return calibration_;
  }

  [[nodiscard]] std::shared_ptr<talos::TalosReader> reader() const noexcept
  {
    return reader_;
  }

private:
  void buildCalibration();

  std::shared_ptr<talos::TalosReader> reader_;
  talos::TalosSimConfig config_;
  std::optional<CameraCalibration> calibration_;
};

}  // namespace L1Sensor
