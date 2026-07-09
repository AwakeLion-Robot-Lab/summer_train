#pragma once

namespace L5Control {

struct SerialCommand {
  double yaw = 0.0;
  double pitch = 0.0;
  bool shoot = false;
};

}  // namespace L5Control
