#include "l4_planning/planner.hpp"

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/latency_compensator.hpp"
#include "l4_planning/predictor.hpp"
#include "l4_planning/types.hpp"

#include <optional>

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
