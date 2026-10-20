#pragma once

#include "l1_sensor/talos/shm_layout.hpp"
#include "l5_control/serial_command.hpp"

namespace L1Sensor::talos {

GimbalCmd encodeGimbalCommand(
  const L5Control::SerialCommand& command,
  double distance_m,
  bool shoot_enable) noexcept;

struct SimulatorSetpoint {
  double yaw_ros{};    // rad，ROS/project barrel yaw
  double pitch_ros{};  // rad，与项目一致：正值低头，负值抬头
};

// 复现当前 Rust process_subscription 的解码公式，用于协议金样测试。
SimulatorSetpoint decodeSimulatorSetpoint(const GimbalCmd& command) noexcept;

} // namespace L1Sensor::talos
