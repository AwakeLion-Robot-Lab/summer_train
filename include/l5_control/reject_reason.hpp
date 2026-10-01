#pragma once

#include <string>

namespace L5Control {

// 每条原因只对应一种情况，一帧里同一件事不会记两遍。
enum class RejectReason {
  ShootDisabled,
  NoTarget,
  NotTracking,
  TempLost,
  // 下面两条一一对应 L4 的 PlanError：规划失败，本帧没有新的瞄准角。
  // 命中时刻没有装甲板落在可击打窗口内（小陀螺的正常间歇）。
  OutOfWindow,
  BallisticFailed,
  // 拿不到云台实际姿态，算不了瞄准误差。
  NoPose,
  // 云台还没转到位：实际角与规划角之差超过了装甲板在该距离上张开的角度。
  // 与 OutOfWindow 分开记，前者是目标的问题，这里是云台的问题。
  AimError,
  CommandJump
};

std::string toString(RejectReason reason);

}  // namespace L5Control
