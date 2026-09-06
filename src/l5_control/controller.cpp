#include "l5_control/controller.hpp"

#include <algorithm>
#include <cmath>

namespace L5Control {

 

 SerialCommand Controller::makeCommand(
  const L4Planning::AimPlan& plan,
  const FireDecision& decision,
  const L1Sensor::RobotState& robot_state) const
{
  SerialCommand command{};

  // 输出边界必须始终生成有限数；上游状态异常时退回协议安全零位。
  const double hold_yaw =
    std::isfinite(robot_state.rpy.yaw) ? robot_state.rpy.yaw : 0.0;
  const double hold_pitch =
    std::isfinite(robot_state.rpy.pitch) ? robot_state.rpy.pitch : 0.0;

  const auto hold = [&command, hold_yaw, hold_pitch]() {
    command.yaw = hold_yaw;
    command.pitch = hold_pitch;
    command.shoot = false;
    return command;
  };

  // 规划无效：保持当前云台角度，并且禁止开火。
  if (!plan.valid) {
    return hold();
  }

  const auto has_reason = [&decision](RejectReason reason) {
    return std::find(decision.reasons.begin(), decision.reasons.end(), reason) !=
           decision.reasons.end();
  };
  if (has_reason(RejectReason::NonFinite) ||
      has_reason(RejectReason::PlanInvalid) ||
      has_reason(RejectReason::RobotStateStale) ||
      has_reason(RejectReason::GimbalPoseStale) ||
      has_reason(RejectReason::OutOfRange) ||
      has_reason(RejectReason::CommandJump)) {
    return hold();
  }

  // MPC 模式使用 samples 中的当前控制点。
  if (plan.using_MPC) {
    // MPC 模式却没有控制点，按无效规划处理。
    if (plan.samples.empty()) {
      return hold();
    }

    command.yaw = plan.samples.front().yaw;
    command.pitch = plan.samples.front().pitch;
  } else {
    // 非 MPC 模式直接使用 Plan 中的瞄准角度。
    command.yaw = plan.yaw;
    command.pitch = plan.pitch;
  }

  if (!std::isfinite(command.yaw) || !std::isfinite(command.pitch)) {
    return hold();
  }

  // Controller 不再重新判断开火条件，直接采用 L5 结果。
  command.shoot = decision.shoot;

  return command;
}

}  // namespace L5Control
