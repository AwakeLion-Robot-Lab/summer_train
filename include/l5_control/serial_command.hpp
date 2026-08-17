#pragma once

namespace L5Control {

struct SerialCommand {
  double yaw = 0.0;    // 世界系绝对角，rad
  double pitch = 0.0;  // 世界系绝对角，rad
  bool shoot = false;  // 本条命令是否置开发射位
};

}  // namespace L5Control
