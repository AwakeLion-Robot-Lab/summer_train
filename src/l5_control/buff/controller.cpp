#include "l5_control/buff/controller.hpp"

namespace L5Control::Buff {

std::optional<SerialCommand> Controller::update(
  const L4Planning::Buff::Plan& plan)
{
  if (!plan.valid) return safeHold();
  SerialCommand command;
  command.yaw = plan.yaw;
  command.pitch = plan.pitch;
  command.shoot = config_.shoot_enable && plan.algorithm_shoot;
  last_command_ = command;
  return command;
}

std::optional<SerialCommand> Controller::safeHold() const
{
  // 尚无有效规划时不存在可保持的角度。不能用零值代替：L5 命令是世界系
  // 绝对角，云台当前若朝向 ±180°，下发 yaw=0 会造成一次猛回头。
  if (!last_command_) return std::nullopt;
  SerialCommand command = *last_command_;
  command.shoot = false;
  return command;
}

void Controller::reset() noexcept
{
  last_command_.reset();
}

}  // namespace L5Control::Buff
