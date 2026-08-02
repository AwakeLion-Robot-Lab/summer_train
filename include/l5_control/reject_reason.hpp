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
  OutsideHitWindow,
  ArmorSwitching,
  CommandJump,
  MissingCalibration,
  NonFinite,
  Unstable
};

std::string toString(RejectReason reason);

}  // namespace L5Control
