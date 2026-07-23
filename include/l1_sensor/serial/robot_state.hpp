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

struct Orientation
{
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};


struct RobotState {

  Orientation rpy;

  // 云台当前运动状态，单位依次为 rad/s 和 rad/s^2。
  double yaw_rate = 0.0;
  double pitch_rate = 0.0;
  double yaw_acceleration = 0.0;
  double pitch_acceleration = 0.0;

  double bullet_speed = 0.0;
  double heat = 0.0;
  EnemyColor enemy_color = EnemyColor::Unknown;
  WorkMode mode = WorkMode::Idle;
  //这个时间辍是收到消息打上的，所以忽略了通信延迟(未知)，下位的运行延迟(2～3ms)
  std::chrono::steady_clock::time_point timestamp{};
};

std::string toString(EnemyColor color);
std::string toString(WorkMode mode);

}  // namespace L1Sensor
