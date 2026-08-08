#include "l5_control/controller.hpp"

#include <cmath>

namespace L5Control {

std::optional<SerialCommand> Controller::makeCommand(
  const L4Planning::Plan& plan, const FireDecision& decision) const
{
  // 规划失败或角度非有限时不下发，交给下位机保持上一状态。
  if (!plan.valid || !std::isfinite(plan.yaw) || !std::isfinite(plan.pitch)) {
    return std::nullopt;
  }

  // shoot 只能来自 FireDecision。曾经这里直接接的是 plan.valid，那样会绕过
  // FireConfig::shoot_enable、命中角度判据和全部 RejectReason——等于"规划成功
  // 就开火"，验收前的安全闸门形同虚设。
  return SerialCommand{plan.yaw, plan.pitch, decision.shoot};
}

}  // namespace L5Control
