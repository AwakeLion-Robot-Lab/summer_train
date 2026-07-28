#include "l5_control/controller.hpp"

namespace L5Control {

 SerialCommand  makeCommand(
    const L4Planning::AimPlan& plan,
    const FireDecision& decision,
    const L1Sensor::RobotState& robot_state) const
{
  return SerialCommand{plan.yaw, plan.pitch, plan.valid};
}

}  // namespace L5Control
