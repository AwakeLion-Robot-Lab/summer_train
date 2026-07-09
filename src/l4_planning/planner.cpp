#include "l4_planning/planner.hpp"

namespace L4Planning {

AimPlan Planner::plan(
  const std::optional<L3Estimation::TargetState>& target,
  const L1Sensor::RobotState& robot_state)
{
  (void)robot_state;
  AimPlan plan;
  plan.valid = target.has_value();
  return plan;
}

}  // namespace L4Planning
