#pragma once

#include "l4_planning/state.hpp"
#include "l4_planning/types.hpp"

namespace L1Sensor {
struct RobotState;
}

namespace L4Planning {

struct PlannerContext;

[[nodiscard]] bool validPlannerConfig(
  const PlannerConfig& config) noexcept;

[[nodiscard]] bool validArmorScoreWeights(
  const ArmorScoreWeights& weights) noexcept;

[[nodiscard]] bool validFacingAngleThresholds(
  double good_angle_degree,
  double bad_angle_degree) noexcept;

[[nodiscard]] bool validPlanningContext(
  const PlannerContext& context) noexcept;

[[nodiscard]] bool validRobotStateForPlanning(
  const L1Sensor::RobotState& robot_state) noexcept;

}  // namespace L4Planning
