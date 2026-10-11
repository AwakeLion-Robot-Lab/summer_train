#pragma once

namespace L5Control {

struct SerialCommand {
  double yaw = 0.0;    // 世界系绝对角，rad
  double pitch = 0.0;  // 世界系绝对角，rad
  bool shoot = false;  // 本条命令是否置开发射位

  // 下行可选前馈，是否发送由串口配置决定。
  double yaw_velocity = 0.0;
  double yaw_acceleration = 0.0;
  double pitch_velocity = 0.0;
  double pitch_acceleration = 0.0;
};

}  // namespace L5Control
