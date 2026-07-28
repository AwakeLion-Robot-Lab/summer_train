#include "l5_control/fire_decision.hpp"

namespace L5Control {

bool evaluateFire(const L4Planning::AimPlan& plan)
{
  return plan.valid;
}

bool shouldFire(const L4Planning::AimPlan& plan)
{
  return evaluateFire(plan);
}

}  // namespace L5Control
