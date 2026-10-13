#pragma once

#include <string>

namespace L5Control {

/**
 * @brief 不开火的原因
 *
 * 每条原因只对应一种情况，一帧里同一件事不会记两遍。
 */
enum class RejectReason {
  ShootDisabled,    ///< 总开火开关 shoot_enable 关着
  NoTarget,         ///< 没有目标
  NotTracking,      ///< 目标还在 Detecting 或已 Lost
  TempLost,         ///< 短时丢失，状态全靠外推
  /// L4 OutOfWindow：命中时刻没有装甲板落在可击打窗口内（小陀螺的正常间歇）。
  /// 这条和下一条一一对应 L4 的 PlanError：规划失败，本帧没有新的瞄准角。
  OutOfWindow,
  BallisticFailed,  ///< L4 BallisticFailed：弹道无解
  NoPose,           ///< 拿不到云台实际姿态，算不了瞄准误差。
  /// 云台还没转到位：实际角与规划角之差超过了装甲板在该距离上张开的角度。
  /// 与 OutOfWindow 分开记，前者是目标的问题，这里是云台的问题。
  AimError,
  CommandJump       ///< yaw 命令相对上一条跳变超过 command_jump_threshold
};

/**
 * @brief 拒绝原因的名字，如 "out_of_window"
 * @param reason 拒绝原因
 * @return snake_case 名字，供日志和叠加层使用
 */
std::string toString(RejectReason reason);

}  // namespace L5Control
