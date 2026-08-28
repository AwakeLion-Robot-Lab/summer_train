#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"

#include <chrono>
#include <cstdint>

// L3 里与目标类型无关的东西。装甲板专有的类别、观测和配置在 armor/types.hpp，
// 符的在 buff/ 下——顶层不认识任何具体目标。
namespace L3Estimation {

// L3 中的时间戳统一使用单调时钟，避免系统时间校准造成负时间差。
using TimePoint = std::chrono::steady_clock::time_point;

// 四态跟踪机的形状与目标类型无关，装甲板和符共用同一组状态。
enum class TrackState : std::uint8_t {
  Lost,       // 当前没有可用目标
  Detecting,  // 已发现目标，等待连续帧确认
  Tracking,   // 稳定跟踪
  TempLost    // 短时丢失，继续输出预测状态
};

// 相机内参及静态机械外参由 L1 持有，L3 只补充时间同步状态。
struct AimCalibration {
  // 内参、畸变参数以及 camera -> barrel 的静态外参。
  L1Sensor::CameraCalibration camera;
  // 图像曝光时刻与枪管姿态已经完成时间对齐。
  bool time_sync_ok{false};

  // 这里只检查 PnP 所需矩阵是否存在，矩阵数值由 PnpSolver 进一步验证。
  [[nodiscard]] bool intrinsicsOk() const noexcept
  {
    return !camera.camera_matrix.empty() &&
           !camera.distortion_coefficients.empty();
  }

  [[nodiscard]] bool trackingReady() const noexcept
  {
    return intrinsicsOk() && camera.barrelExtrinsicsReady() && time_sync_ok;
  }

  [[nodiscard]] bool fireReady() const noexcept
  {
    return trackingReady();
  }
};
}  // namespace L3Estimation
