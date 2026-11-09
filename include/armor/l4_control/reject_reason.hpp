#pragma once

#include <string>

namespace L5Control {

enum class RejectReason {
  ShootDisabled,
  NoTarget,
  NotTracking,
  TempLost,
  PlanInvalid,
  BallisticInvalid,
  BadBulletSpeed,
  // 延迟链缺少实车标定段，落点会系统性偏早，只跟随不开火。
  DelayNotCalibrated,
  // 命中时刻没有装甲板落在可击发窗口内（小陀螺的正常间歇）。
  OutsideHitWindow,
  // 实际角相对当前瞄点指令的分轴误差超过允许容差；MPC 使用当帧执行采样。
  // 与 OutsideHitWindow 分开记，前者是目标的问题，后者是云台的问题。
  AimError,
  CommandJump,
  // 区域 MPC 当前指令与运动装甲板无安全区交点，例如两板之间的过渡。
  MpcAimOutsidePlate,
  NonFinite
};

std::string toString(RejectReason reason);

}  // namespace L5Control
