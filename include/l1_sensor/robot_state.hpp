#pragma once

#include <chrono>
#include <string>

namespace L1Sensor {

enum class EnemyColor {
  Red,
  Blue,
  Unknown
};

enum class WorkMode {
  Idle,
  AutoAim,
  SmallBuff,
  BigBuff,
  Outpost
};

struct RobotState {
  double yaw = 0.0;
  double pitch = 0.0;
  double bullet_speed = 0.0;
  double heat = 0.0;
  EnemyColor enemy_color = EnemyColor::Unknown;
  WorkMode mode = WorkMode::Idle;
  std::chrono::steady_clock::time_point timestamp{};
};

std::string toString(EnemyColor color);
std::string toString(WorkMode mode);

}  // namespace L1Sensor
