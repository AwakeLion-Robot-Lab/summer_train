#include "rune/f4-planning/planning.hpp"

#include "rune/f4-planning/ballistic_solver.hpp"

#include <Eigen/Geometry>

#include <cmath>
#include <numbers>

namespace Rune {

Command PlanningStage::plan(const Motion& motion, Mode mode,
                             const ControlInput& control) {
  Command command;
  command.mode = mode;
  command.status = Status::MotionNotReady;
  if (!motion.ready || (mode == Mode::Big && !motion.phase_delta_rad))
    return command;
  if (motion.pose.captured_at == TimePoint{} ||
      control.command_at == TimePoint{}) return command;

  const double age = std::chrono::duration<double>(
    control.command_at - motion.pose.captured_at).count();
  if (!std::isfinite(age) || age < 0 || age > config_.max_observation_age_s ||
      !std::isfinite(control.fire_delay_s) || control.fire_delay_s < 0)
    return command;

  const auto& pose = motion.pose;
  const Eigen::Vector3d radius = pose.target_barrel - pose.center_barrel;
  if (radius.norm() <= 0 || !pose.normal_barrel.allFinite() ||
      pose.normal_barrel.norm() < 0.9 ||
      !std::isfinite(control.bullet_speed_m_s) ||
      control.bullet_speed_m_s <= 0) {
    command.status = Status::BallisticFailed;
    return command;
  }
  double flight_time = radius.norm() / control.bullet_speed_m_s;
  if (!std::isfinite(flight_time) || flight_time <= 0) {
    command.status = Status::BallisticFailed;
    return command;
  }

  std::optional<detail::BallisticSolution> ballistic;
  bool converged = false;
  for (int iteration = 0; iteration < 12; ++iteration) {
    const double horizon = age + control.fire_delay_s + flight_time;
    const auto phase_delta = motion.phase_delta_rad
      ? motion.phase_delta_rad(horizon)
      : std::optional<double>(motion.angular_speed_rad_s * horizon);
    if (!phase_delta || !std::isfinite(*phase_delta)) {
      command.status = Status::MotionNotReady;
      return command;
    }
    const Eigen::Vector3d future = pose.center_barrel +
      Eigen::AngleAxisd(*phase_delta,
                        pose.normal_barrel.normalized()) * radius;
    ballistic = detail::solveBallistic(future, control.bullet_speed_m_s,
                                       config_.gravity_m_s2);
    if (!ballistic) break;
    const double change = std::abs(ballistic->flight_time - flight_time);
    flight_time = ballistic->flight_time;
    if (change < 0.0002) { converged = true; break; }
  }
  if (!ballistic || !converged) {
    command.status = Status::BallisticFailed;
    return command;
  }
  command.yaw_rad = ballistic->yaw;
  command.pitch_rad = ballistic->pitch;
  command.flight_time_s = ballistic->flight_time;
  command.found = true;
  command.status = Status::AimNotReady;
  if (pose.activated || !std::isfinite(control.actual_yaw_rad) ||
      !std::isfinite(control.actual_pitch_rad)) return command;

  const double yaw_error = std::remainder(
    command.yaw_rad - control.actual_yaw_rad, 2 * std::numbers::pi);
  if (std::abs(yaw_error) > config_.max_yaw_error_rad ||
      std::abs(command.pitch_rad - control.actual_pitch_rad) >
        config_.max_pitch_error_rad) return command;
  if (!config_.shoot_enable) return command;
  if (fired_) {
    command.status = Status::AlreadyFired;
    return command;
  }
  fired_ = true;
  command.status = Status::Ready;
  command.fire = true;
  return command;
}

}  // namespace Rune
