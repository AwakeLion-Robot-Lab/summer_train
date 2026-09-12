#pragma once

namespace L5Control {

struct SerialCommand {
  double yaw = 0.0;    // 世界系绝对角，rad
  double pitch = 0.0;  // 世界系绝对角，rad
  bool shoot = false;  // 本条命令是否置开发射位

  // 规划出的一阶、二阶前馈量，供下位机叠加到自己的控制环上（电控用它做 LQR
  // 解算）。没开轨迹规划时恒为 0；下行帧是否真的携带它们由 serial_config 的
  // command_format 决定，与这里是否有值无关。
  double yaw_velocity = 0.0;       // rad/s
  double yaw_acceleration = 0.0;   // rad/s^2
  double pitch_velocity = 0.0;     // rad/s
  double pitch_acceleration = 0.0; // rad/s^2
};

}  // namespace L5Control
