#include "l5_control/reject_reason.hpp"

namespace L5Control {

std::string toString(RejectReason reason)
{
  switch (reason) {
    case RejectReason::ShootDisabled:
      return "shoot_disabled";
    case RejectReason::NoTarget:
      return "no_target";
    case RejectReason::NotTracking:
      return "not_tracking";
    case RejectReason::TempLost:
      return "temp_lost";
    case RejectReason::PlanInvalid:
      return "plan_invalid";
    case RejectReason::BallisticInvalid:
      return "ballistic_invalid";
    case RejectReason::BadBulletSpeed:
      return "bad_bullet_speed";
    case RejectReason::DelayNotCalibrated:
      return "delay_not_calibrated";
    case RejectReason::OutsideHitWindow:
      return "outside_hit_window";
    case RejectReason::AimError:
      return "aim_error";
    case RejectReason::CommandJump:
      return "command_jump";
    case RejectReason::NonFinite:
      return "non_finite";
    case RejectReason::Blending:
      return "blending";
    default:
      return "unknown";
  }
}

}  // namespace L5Control
