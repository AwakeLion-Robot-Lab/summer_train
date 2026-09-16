#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"

#include <chrono>
#include <cstdint>

// L3 里与目标类型无关的公共定义。装甲板专有的类别、观测和配置在
// armor/types.hpp，符的在 buff/ 下，顶层不认识任何具体目标。
namespace L3Estimation {

// 时间戳统一用单调时钟，系统时间被校准时也不会出现负的帧间隔。
using TimePoint = std::chrono::steady_clock::time_point;

// 四态跟踪状态机，装甲板和符共用。转移规则见 association.hpp 的 updateFsm。
enum class TrackState : std::uint8_t {
  Lost,       // 当前没有可用目标
  Detecting,  // 已发现目标，等待连续帧确认
  Tracking,   // 稳定跟踪
  TempLost    // 短时丢失，继续输出预测状态
};

// L3 求解需要的标定。内参和静态外参由 L1 持有，这里只补一个时间同步标志。
struct AimCalibration {
  // 内参、畸变参数以及 camera -> barrel 的静态外参。
  L1Sensor::CameraCalibration camera;
  // 图像曝光时刻与枪管姿态已经完成时间对齐。
  bool time_sync_ok{false};

  // 只检查 PnP 要用的两个矩阵在不在，数值是否合法由 PnpSolver::setCalibration 验。
  bool intrinsicsOk() const noexcept
  {
    return !camera.camera_matrix.empty() &&
           !camera.distortion_coefficients.empty();
  }

  bool trackingReady() const noexcept
  {
    return intrinsicsOk() && camera.barrelExtrinsicsReady() && time_sync_ok;
  }

  bool fireReady() const noexcept
  {
    return trackingReady();
  }
};
}  // namespace L3Estimation
