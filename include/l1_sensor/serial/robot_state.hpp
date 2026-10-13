#pragma once

#include <chrono>
#include <string>

namespace L1Sensor {

/**
 * @brief 下位机告知的敌方颜色
 *
 * Unknown 表示不是有效目标，用来防误伤友军。
 */
enum class EnemyColor {
  Red,
  Blue,
  Unknown
};

/**
 * @brief 下位机给出的工作模式，决定 runtime 每帧走哪条分支
 */
enum class WorkMode {
  Idle,
  AutoAim,
  SmallBuff,
  BigBuff,
  Outpost
};

/**
 * @brief 云台欧拉角，单位 rad
 */
struct Orientation
{
  /// 横滚角
  double roll = 0.0;

  /// 俯仰角
  double pitch = 0.0;

  /// 偏航角
  double yaw = 0.0;
};


/**
 * @brief 下位机回传的一帧机器人状态
 */
struct RobotState {

  /// 云台姿态（IMU 系）
  Orientation rpy;

  /// 弹速，m/s；下位机没发时为 0
  double bullet_speed = 0.0;

  /// 敌方颜色
  EnemyColor enemy_color = EnemyColor::Unknown;

  /// 工作模式
  WorkMode mode = WorkMode::Idle;

  /// 这个时间辍是收到消息打上的，所以忽略了通信延迟(未知)，下位的运行延迟(2～3ms)
  std::chrono::steady_clock::time_point timestamp{};
};

/**
 * @brief 敌方颜色的可读名字
 * @param color 敌方颜色
 * @return 用于日志的名字
 */
std::string toString(EnemyColor color);

/**
 * @brief 工作模式的可读名字
 * @param mode 工作模式
 * @return 用于日志的名字
 */
std::string toString(WorkMode mode);

}  // namespace L1Sensor
