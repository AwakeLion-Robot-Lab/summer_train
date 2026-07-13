#include "l1_sensor/serial/robot_state.hpp"

namespace L1Sensor {

std::string toString(EnemyColor color)
{
  switch (color) {
    case EnemyColor::Red:
      return "red";
    case EnemyColor::Blue:
      return "blue";
    case EnemyColor::Unknown:
    default:
      return "unknown";
  }
}

std::string toString(WorkMode mode)
{
  switch (mode) {
    case WorkMode::Idle:
      return "idle";
    case WorkMode::AutoAim:
      return "auto_aim";
    case WorkMode::SmallBuff:
      return "small_buff";
    case WorkMode::BigBuff:
      return "big_buff";
    case WorkMode::Outpost:
      return "outpost";
    default:
      return "unknown";
  }
}

}  // namespace L1Sensor
