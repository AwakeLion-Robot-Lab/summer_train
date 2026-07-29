#include "l5_control/fire_decision.hpp"

namespace L5Control {

bool shouldFire(const L4Planning::AimPlan& plan)
{
  return plan.valid && plan.fire_permitted && !plan.armor_switching;
}

}  // namespace L5Control
