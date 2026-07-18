#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/types.hpp"

#include <optional>

namespace L4Planning {

class Planner {
public:
  AimPlan plan(const std::optional<L3Estimation::TargetState>& target, const L1Sensor::RobotState& robot_state);
};

}  // namespace L4Planning
