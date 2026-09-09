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
  // 云台还没转到位：实际角与规划角之差超过了装甲板在该距离上张开的角度。
  // 与 OutsideHitWindow 分开记，前者是目标的问题，后者是云台的问题。
  AimError,
  CommandJump,
  NonFinite,
  // 切板过渡段进行中：命令是五次多项式上的值，**故意**偏离射击轨迹，此时
  // 云台跟得再好也打不中。新枚举一律追加在末尾——遥测按整数值画曲线，
  // 中间插一个会让历史记录整体错位。
  Blending
};

std::string toString(RejectReason reason);

}  // namespace L5Control
