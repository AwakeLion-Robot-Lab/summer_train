#include "l5_control/controller.hpp"

namespace L5Control {

SerialCommand Controller::makeCommand(const L4Planning::AimPlan& plan) const
{
  SerialCommand command;
  if (!plan.valid || (plan.using_MPC && plan.samples.empty())) {
    return command;
  }

  if (plan.using_MPC) {
    const L4Planning::AimSample& current = plan.samples.front();
    command.yaw = current.yaw;
    command.pitch = current.pitch;
    command.yaw_rate = current.yaw_rate;
    command.pitch_rate = current.pitch_rate;
    command.yaw_acceleration = current.yaw_acceleration;
    command.pitch_acceleration = current.pitch_acceleration;
  } else {
    command.yaw = plan.yaw;
    command.pitch = plan.pitch;
    command.yaw_rate = plan.yaw_rate;
    command.pitch_rate = plan.pitch_rate;
    command.yaw_acceleration = plan.yaw_acceleration;
    command.pitch_acceleration = plan.pitch_acceleration;
  }
  command.shoot = plan.fire_permitted;
  return command;
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
    command.yaw_rate = plan.samples.front().yaw_rate;
    command.pitch_rate = plan.samples.front().pitch_rate;
    command.yaw_acceleration = plan.samples.front().yaw_acceleration;
    command.pitch_acceleration = plan.samples.front().pitch_acceleration;
  } else {
    command.yaw = plan.yaw;
    command.pitch = plan.pitch;
    command.yaw_rate = plan.yaw_rate;
    command.pitch_rate = plan.pitch_rate;
    command.yaw_acceleration = plan.yaw_acceleration;
    command.pitch_acceleration = plan.pitch_acceleration;
  }

  command.shoot = decision.shoot;
  return command;
}

}  // namespace L5Control
