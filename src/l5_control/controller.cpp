#include "l5_control/controller.hpp"

#include "l6_telemetry/math.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace L5Control {
namespace {

constexpr double kDefaultCommandJump = 10.0 * std::numbers::pi / 180.0;

std::optional<Eigen::Vector2d> commandAngles(
  const L4Planning::AimPlan& plan) noexcept
{
  if (!plan.valid || (plan.using_MPC && plan.samples.empty())) {
    return std::nullopt;
  }
  if (plan.using_MPC) {
    return Eigen::Vector2d{plan.samples.front().yaw, plan.samples.front().pitch};
  }
  return Eigen::Vector2d{plan.yaw, plan.pitch};
}

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
  const L4Planning::AimPlan& plan,
  const std::optional<Eigen::Quaterniond>& actual_pose,
  bool bullet_speed_valid)
{
  std::optional<Eigen::Vector2d> actual_angles;
  if (actual_pose && actual_pose->coeffs().allFinite()) {
    const Eigen::Vector3d ypr = L6Telemetry::eulers(
      actual_pose->toRotationMatrix(), 2, 1, 0);
    if (ypr.allFinite()) {
      actual_angles = ypr.head<2>();
    }
  }

  const auto command_angles = commandAngles(plan);
  FireInput input;
  input.target = target;
  input.track_state = track_state;
  input.plan = plan;
  input.actual_yaw = actual_angles
    ? (*actual_angles)[0]
    : std::numeric_limits<double>::quiet_NaN();
  input.actual_pitch = actual_angles
    ? (*actual_angles)[1]
    : std::numeric_limits<double>::quiet_NaN();
  input.command_jump = command_angles && last_command_ &&
    std::abs(L6Telemetry::limit_rad(
      (*command_angles)[0] - last_command_->yaw)) > command_jump_threshold_;
  input.bullet_speed_valid = bullet_speed_valid;

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
  const L4Planning::AimPlan& plan,
  const FireDecision& decision) const
{
  const auto angles = commandAngles(plan);
  if (!angles) {
    return std::nullopt;
  }
  SerialCommand command{(*angles)[0], (*angles)[1], decision.shoot};
  if (plan.using_MPC && !plan.samples.empty()) {
    const auto &sample = plan.samples.front();
    command.yaw_velocity = sample.yaw_rate;
    command.yaw_acceleration = sample.yaw_acceleration;
    command.pitch_velocity = sample.pitch_rate;
    command.pitch_acceleration = sample.pitch_acceleration;
  } else {
    command.yaw_velocity = plan.yaw_rate;
    command.yaw_acceleration = plan.yaw_acceleration;
    command.pitch_velocity = plan.pitch_rate;
    command.pitch_acceleration = plan.pitch_acceleration;
  }
  return command;
}

}  // namespace L5Control
