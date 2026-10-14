#pragma once

#include "sensor/serial/serial_command.hpp"

namespace L5Control {

// Compatibility name for armor control callers. The shared wire command is
// owned by L1Sensor so camera/serial users do not depend on armor control.
using SerialCommand = L1Sensor::SerialCommand;
}  // namespace L5Control
