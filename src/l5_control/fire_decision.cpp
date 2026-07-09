#include "l5_control/fire_decision.hpp"

namespace L5Control {

bool shouldFire(const L4Planning::AimPlan& plan)
{
  return plan.valid;
}

}  // namespace L5Control
