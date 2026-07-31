#include "l5_control/controller.hpp"

namespace L5Control {

SerialCommand Controller::makeCommand(
  const L4Planning::AimPlan& plan,
  const FireDecision& decision,
  const L1Sensor::RobotState& robot_state) const
{
  SerialCommand command{};

  // 规划无效时保持云台当前角度，并且禁止开火。
  if (!plan.valid) {
    command.yaw = robot_state.rpy.yaw;
    command.pitch = robot_state.rpy.pitch;
    command.shoot = false;
    return command;
  }

  if (plan.using_MPC) {
    // MPC 模式没有当前控制点时，按无效规划处理。
    if (plan.samples.empty()) {
      command.yaw = robot_state.rpy.yaw;
      command.pitch = robot_state.rpy.pitch;
      command.shoot = false;
      return command;
    }

    command.yaw = plan.samples.front().yaw;
    command.pitch = plan.samples.front().pitch;
  } else {
    command.yaw = plan.yaw;
    command.pitch = plan.pitch;
  }

  // FireEvaluator 已经判断开火条件，Controller 只负责组装命令。
  command.shoot = decision.shoot;
  return command;
}

}  // namespace L5Control
