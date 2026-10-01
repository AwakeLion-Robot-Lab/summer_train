#include "l5_control/reject_reason.hpp"

namespace L5Control {

std::string toString(RejectReason reason)
{
  switch (reason) {
    case RejectReason::ShootDisabled: return "shoot_disabled";
    case RejectReason::NoTarget: return "no_target";
    case RejectReason::NotTracking: return "not_tracking";
    case RejectReason::TempLost: return "temp_lost";
    case RejectReason::OutOfWindow: return "out_of_window";
    case RejectReason::BallisticFailed: return "ballistic_failed";
    case RejectReason::NoPose: return "no_pose";
    case RejectReason::AimError: return "aim_error";
    case RejectReason::CommandJump: return "command_jump";
  }
  return "unknown";
}

}  // namespace L5Control
