#pragma once

#include "l1_sensor/robot_state.hpp"
#include "l3_estimation/target_estimator.hpp"

#include <optional>

namespace L4Planning {

struct AimPlan {
  double yaw = 0.0;
  double pitch = 0.0;
  double fly_time = 0.0;
  bool valid = false;
};

class Planner {
public:
  AimPlan plan(const std::optional<L3Estimation::TargetState>& target, const L1Sensor::RobotState& robot_state);
};

}  // namespace L4Planning
