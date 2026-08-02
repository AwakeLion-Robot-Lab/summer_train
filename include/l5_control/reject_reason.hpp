#pragma once

#include <string>

namespace L5Control {

enum class RejectReason {
  None,
  ShootDisabled,
  ParametersNotReady,
  AutoAimDisabled,
  NoTarget,
  NotTracking,
  TempLost,
  PlanInvalid,
  BallisticInvalid,
  RobotStateStale,
  GimbalPoseStale,
  BadBulletSpeed,
  HeatLimit,
  OutOfRange,
  
  OutsideHitWindow,
  ArmorSwitching,
  CommandJump,
  MissingCalibration,
  NonFinite,
  Unstable
};

std::string toString(RejectReason reason);

}  // namespace L5Control
