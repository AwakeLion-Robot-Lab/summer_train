#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/fire_decision.hpp"
#include "l5_control/serial_command.hpp"

namespace L5Control {

class Controller {
public:
  [[nodiscard]] SerialCommand makeCommand(
    const L4Planning::AimPlan& plan) const;
  [[nodiscard]] SerialCommand makeCommand(
    const L4Planning::AimPlan& plan,
    const FireDecision& decision,
    const L1Sensor::RobotState& robot_state) const;
};

}  // namespace L5Control
