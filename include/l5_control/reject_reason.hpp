#pragma once

#include <string>

namespace L5Control {

enum class RejectReason {
  None,
  ShootDisabled,
  NoTarget,
  NotTracking,
  TempLost,
  PlanInvalid,
  BallisticInvalid,
  RobotStateStale,
  GimbalPoseStale,
  BadBulletSpeed,
  OutOfRange,
  HeatLimit,
  // 命中时刻没有装甲板落在可击发窗口内（小陀螺的正常间歇）。
  OutsideHitWindow,
  // 云台还没转到位：实际角与规划角之差超过了装甲板在该距离上张开的角度。
  // 与 OutsideHitWindow 分开记，前者是目标的问题，后者是云台的问题。
  AimError,
  ArmorSwitching,
  CommandJump,
  MissingCalibration,
  NonFinite,
  Unstable
};

std::string toString(RejectReason reason);

}  // namespace L5Control
