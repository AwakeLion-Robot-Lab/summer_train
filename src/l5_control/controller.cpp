#include "l5_control/controller.hpp"

namespace L5Control {

SerialCommand Controller::makeCommand(const L4Planning::AimPlan& plan) const
{
  return SerialCommand{plan.yaw, plan.pitch, plan.fire_permitted};
}

SerialCommand Controller::makeCommand(
  const L4Planning::AimPlan& plan,
  const FireDecision& decision,
  const L1Sensor::RobotState& robot_state) const
{
  SerialCommand command{};

  if (!plan.valid) {
    command.yaw = robot_state.rpy.yaw;
    command.pitch = robot_state.rpy.pitch;
    command.shoot = false;
    return command;
  }

  if (plan.using_MPC) {
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

  command.shoot = decision.shoot;
  return command;
}

}  // namespace L5Control
