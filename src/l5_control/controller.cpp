#include "l5_control/controller.hpp"

namespace L5Control {

SerialCommand Controller::makeCommand(const L4Planning::AimPlan& plan) const
{
  const bool shoot =
    plan.valid && plan.fire_permitted && !plan.armor_switching;
  return SerialCommand{plan.yaw, plan.pitch, shoot};
}

}  // namespace L5Control
