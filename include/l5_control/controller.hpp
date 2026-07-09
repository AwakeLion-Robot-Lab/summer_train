#pragma once

#include "l4_planning/planner.hpp"
#include "l5_control/serial_command.hpp"

namespace L5Control {

class Controller {
public:
  SerialCommand makeCommand(const L4Planning::AimPlan& plan) const;
};

}  // namespace L5Control
