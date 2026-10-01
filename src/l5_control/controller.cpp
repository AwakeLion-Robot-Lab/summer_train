#include "l5_control/controller.hpp"

#include "l6_telemetry/math.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace L5Control {

Controller::Controller(FireConfig fire_config) noexcept
: fire_decider_(std::move(fire_config))
{
}

std::optional<SerialCommand> Controller::update(
  const std::optional<L3Estimation::EskfTarget>& target,
  L3Estimation::TrackState track_state,
  const L4Planning::Plan& plan,
  const std::optional<Eigen::Quaterniond>& actual_pose)
{
  // 串口给出枪管到世界系的姿态。按 Z-Y-X 分解后只取 yaw、pitch；roll 不参与火控。
  std::optional<Eigen::Vector2d> actual_angles;
  if (actual_pose && actual_pose->coeffs().allFinite()) {
    const Eigen::Vector3d ypr = L6Telemetry::eulers(
      actual_pose->toRotationMatrix(), 2, 1, 0);
    if (ypr.allFinite()) {
      actual_angles = ypr.head<2>();
    }
  }

  // 姿态缺失时写入 NaN，FireDecider 以 no_pose 关火；有效 Plan 仍照常下发跟随角。
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  FireInput input;
  if (target) {
    input.target = target->name;
  }
  input.track_state = track_state;
  input.plan = plan;
  input.actual_yaw = actual_angles ? (*actual_angles)[0] : kNaN;
  input.actual_pitch = actual_angles ? (*actual_angles)[1] : kNaN;
  input.command_jump = plan.valid() && last_command_ &&
    std::abs(L6Telemetry::limit_rad(plan.aim.yaw - last_command_->yaw)) >
      fire_decider_.config().command_jump_threshold;

  last_decision_ = fire_decider_.decide(input);
  std::optional<SerialCommand> command = makeCommand(plan, last_decision_);
  if (command) {
    last_command_ = command;
  } else {
    command = safeHold();
  }

  return command;
}

std::optional<SerialCommand> Controller::safeHold() const
{
  if (!last_command_) {
    return std::nullopt;
  }

  SerialCommand command = *last_command_;
  command.shoot = false;
  return command;
}

std::optional<SerialCommand> Controller::makeCommand(
  const L4Planning::Plan& plan, const FireDecision& decision)
{
  // 规划失败时不下发，交给下位机保持上一状态。
  // 角度的有限性由 Planner 在提交 Plan 前保证，这里不再验一遍。
  if (!plan.valid()) {
    return std::nullopt;
  }

  // yaw/pitch 来自 L4，shoot 只能来自完整的 FireDecision，规划成功本身不代表开火。
  return SerialCommand{plan.aim.yaw, plan.aim.pitch, decision.shoot};
}

}  // namespace L5Control
