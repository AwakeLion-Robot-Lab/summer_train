#pragma once

namespace L5Control {

/**
 * @brief 发给下位机的一条控制命令
 */
struct SerialCommand {
  /// 世界系绝对角，rad
  double yaw = 0.0;

  /// 世界系绝对角，rad
  double pitch = 0.0;

  /// 本条命令是否置开发射位
  bool shoot = false;
};

}  // namespace L5Control
