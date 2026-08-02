#include "l5_control/reject_reason.hpp"

namespace L5Control {

std::string toString(RejectReason reason)
{
  switch (reason) {
    case RejectReason::None:
      return "none";
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
    case RejectReason::RobotStateStale:
      return "robot_state_stale";
    case RejectReason::GimbalPoseStale:
      return "gimbal_pose_stale";
    case RejectReason::BadBulletSpeed:
      return "bad_bullet_speed";
    case RejectReason::OutOfRange:
      return "out_of_range";
    case RejectReason::HeatLimit:
      return "heat_limit";
    case RejectReason::OutsideHitWindow:
      return "outside_hit_window";
    case RejectReason::ArmorSwitching:
      return "armor_switching";
    case RejectReason::CommandJump:
      return "command_jump";
    case RejectReason::MissingCalibration:
      return "missing_calibration";
    case RejectReason::NonFinite:
      return "non_finite";
    case RejectReason::Unstable:
      return "unstable";
    default:
      return "unknown";
  }
}

}  // namespace L5Control
