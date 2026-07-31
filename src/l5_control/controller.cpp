#include "l5_control/controller.hpp"

namespace L5Control {

 

 SerialCommand Controller::makeCommand(
  const L4Planning::AimPlan& plan,
  const FireDecision& decision,
  const L1Sensor::RobotState& robot_state) const
{
  SerialCommand command{};

  // 规划无效：保持当前云台角度，并且禁止开火。
  if (!plan.valid) {
    command.yaw = robot_state.rpy.yaw;
    command.pitch = robot_state.rpy.pitch;
    command.shoot = false;
    return command;
  }

  // MPC 模式使用 samples 中的当前控制点。
  if (plan.using_MPC) {
    // MPC 模式却没有控制点，按无效规划处理。
    if (plan.samples.empty()) {
      command.yaw = robot_state.rpy.yaw;
      command.pitch = robot_state.rpy.pitch;
      command.shoot = false;
      return command;
    }

    command.yaw = plan.samples.front().yaw;
    command.pitch = plan.samples.front().pitch;
  } else {
    // 非 MPC 模式直接使用 Plan 中的瞄准角度。
    command.yaw = plan.yaw;
    command.pitch = plan.pitch;
  }

  // Controller 不再重新判断开火条件，直接采用 L5 结果。
  command.shoot = decision.shoot;

  return command;
}

}  // namespace L5Control
