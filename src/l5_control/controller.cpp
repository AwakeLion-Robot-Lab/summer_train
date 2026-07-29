#include "l5_control/controller.hpp"

namespace L5Control {

SerialCommand Controller::makeCommand(const L4Planning::AimPlan& plan) const
{
  return SerialCommand{plan.yaw, plan.pitch, plan.valid};
}

}  // namespace L5Control
