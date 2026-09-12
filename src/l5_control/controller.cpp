#include "l5_control/controller.hpp"

#include "l6_telemetry/math.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace L5Control {
namespace {

constexpr double kDefaultCommandJump = 10.0 * std::numbers::pi / 180.0;

}  // namespace

Controller::Controller() noexcept
: Controller(FireConfig{}, kDefaultCommandJump)
{
}

Controller::Controller(
  FireConfig fire_config,
  double command_jump_threshold) noexcept
: fire_decider_(std::move(fire_config)),
  command_jump_threshold_(
    std::isfinite(command_jump_threshold) && command_jump_threshold >= 0.0
      ? command_jump_threshold
      : kDefaultCommandJump)
{
}

std::optional<SerialCommand> Controller::update(
  const std::optional<L3Estimation::TrackedTarget>& target,
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

  const bool angles_valid = actual_angles.has_value();

  FireInput input;
  input.target = target;
  input.track_state = track_state;
  input.plan = plan;
  input.actual_yaw = angles_valid
    ? (*actual_angles)[0]
    : std::numeric_limits<double>::quiet_NaN();
  input.actual_pitch = angles_valid
    ? (*actual_angles)[1]
    : std::numeric_limits<double>::quiet_NaN();
  // 姿态缺失时写入 NaN，让 FireDecider 关闭开火；有效 Plan 仍可继续下发跟随角。
  input.command_jump = plan.valid() && last_command_ &&
    std::abs(L6Telemetry::limit_rad(plan.aim.yaw - last_command_->yaw)) >
      command_jump_threshold_;

  const FireDecision decision = fire_decider_.decide(input);
  last_decision_ = decision;
  std::optional<SerialCommand> command = makeCommand(plan, decision);
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
  const L4Planning::Plan& plan, const FireDecision& decision) const
{
  // 规划失败时不下发，交给下位机保持上一状态。
  // 角度的有限性由 Planner 在提交 Plan 前保证，这里不再验一遍。
  if (!plan.valid()) {
    return std::nullopt;
  }

  // yaw/pitch 来自 L4，shoot 只能来自完整的 FireDecision，规划成功本身不代表开火。
  // 前馈量一并带上：帧里发不发由 L1 的 command_format 决定，这里只负责如实转达。
  SerialCommand command;
  command.yaw = plan.aim.yaw;
  command.pitch = plan.aim.pitch;
  command.shoot = decision.shoot;
  command.yaw_velocity = plan.aim.yaw_velocity;
  command.yaw_acceleration = plan.aim.yaw_acceleration;
  command.pitch_velocity = plan.aim.pitch_velocity;
  command.pitch_acceleration = plan.aim.pitch_acceleration;
  return command;
}

}  // namespace L5Control
