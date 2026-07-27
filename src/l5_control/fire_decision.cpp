#include "l5_control/fire_decision.hpp"

#include <cmath>
#include <utility>

namespace L5Control {
namespace {

constexpr double kPi = 3.14159265358979323846;

double angleDifference(double lhs, double rhs) noexcept
{
  return std::remainder(lhs - rhs, 2.0 * kPi);
}

bool isFinite(double value) noexcept
{
  return std::isfinite(value);
}

}  // namespace

FireEvaluator::FireEvaluator(FireConfig config)
  : config_(std::move(config))
{
}

FireDecision FireEvaluator::evaluate(const FireInput& input)
{
  FireDecision decision{};

  double command_yaw = input.plan.yaw;
  double command_pitch = input.plan.pitch;
  bool command_valid = true;

  if (input.plan.using_MPC) {
    if (input.plan.samples.empty()) {
      command_valid = false;
    } else {
      command_yaw = input.plan.samples.front().yaw;
      command_pitch = input.plan.samples.front().pitch;
    }
  }

  if (!isFinite(command_yaw) || !isFinite(command_pitch)) {
    command_valid = false;
    decision.reasons.push_back(RejectReason::NonFinite);
  }

  const bool parameters_ready = config_.parametersReady();
  if (!parameters_ready) {
    decision.reasons.push_back(RejectReason::ParametersNotReady);
  }

  if (!input.calibration_ready) {
    decision.reasons.push_back(RejectReason::MissingCalibration);
  }

  if (input.robot_state.mode == L1Sensor::WorkMode::Idle) {
    decision.reasons.push_back(RejectReason::AutoAimDisabled);
  }

  const auto robot_state_age = input.now - input.robot_state.timestamp;
  if (robot_state_age > config_.max_robot_state_age) {
    decision.reasons.push_back(RejectReason::RobotStateStale);
  }

  if (robot_state_age > config_.max_gimbal_pose_age) {
    decision.reasons.push_back(RejectReason::GimbalPoseStale);
  }

  const auto plan_age = input.now - input.plan.generated_at;
  if (plan_age > config_.max_plan_age || !input.plan.valid || !command_valid) {
    decision.reasons.push_back(RejectReason::PlanInvalid);
  }

  if (!input.target.has_value()) {
    decision.reasons.push_back(RejectReason::NoTarget);
  }

  if (!input.plan.tracking) {
    decision.reasons.push_back(RejectReason::NotTracking);
  }

  if (!input.plan.fire_permitted) {
    decision.reasons.push_back(RejectReason::OutsideHitWindow);
  }

  if (parameters_ready) {
    if (input.robot_state.bullet_speed < *config_.min_bullet_speed ||
        input.robot_state.bullet_speed > *config_.max_bullet_speed) {
      decision.reasons.push_back(RejectReason::BadBulletSpeed);
    }

    if (input.robot_state.heat >= *config_.heat_limit) {
      decision.reasons.push_back(RejectReason::HeatLimit);
    }

    if (command_valid &&
        (command_yaw < *config_.min_yaw || command_yaw > *config_.max_yaw ||
         command_pitch < *config_.min_pitch ||
         command_pitch > *config_.max_pitch)) {
      decision.reasons.push_back(RejectReason::OutOfRange);
    }

    const double target_distance = input.plan.aim_point_barrel.norm();
    const bool target_is_near = target_distance <= *config_.yaw_distance_boundary;

    const double max_yaw_error =
      target_is_near ? *config_.near_max_yaw_error : *config_.far_max_yaw_error;
    const double max_yaw_command_jump =
      target_is_near ? *config_.near_max_yaw_command_jump
                     : *config_.far_max_yaw_command_jump;

    if (command_valid) {
      const double yaw_error = std::abs(
        angleDifference(command_yaw, input.robot_state.rpy.yaw));
      const double pitch_error =
        std::abs(command_pitch - input.robot_state.rpy.pitch);

      if (yaw_error >= max_yaw_error ||
          pitch_error > *config_.max_aim_pitch_error ||
          pitch_error > *config_.max_pitch_error) {
        decision.reasons.push_back(RejectReason::Unstable);
      }

      const bool yaw_command_jump =
        last_command_yaw_.has_value() &&
        std::abs(angleDifference(command_yaw, *last_command_yaw_)) >=
          max_yaw_command_jump;
      const bool pitch_command_jump =
        last_command_pitch_.has_value() &&
        std::abs(command_pitch - *last_command_pitch_) >
          *config_.max_pitch_command_jump;

      if (yaw_command_jump || pitch_command_jump || input.command_jump) {
        decision.reasons.push_back(RejectReason::CommandJump);
      }
    }
  }

  const int current_target_id =
    input.target.has_value() ? input.target->robot_id : input.plan.target_id;
  const int current_armor_id = input.plan.armor_id;

  const bool target_changed =
    (last_target_id_ != -1 && current_target_id != -1 &&
     current_target_id != last_target_id_) ||
    (last_armor_id_ != -1 && current_armor_id != -1 &&
     current_armor_id != last_armor_id_);

  if (target_changed || input.armor_switching) {
    decision.reasons.push_back(RejectReason::ArmorSwitching);
    stable_tracking_frames_ = 0;
    last_switch_time_ = input.now;
  } else if (input.target.has_value() && input.plan.valid && input.plan.tracking) {
    ++stable_tracking_frames_;
  } else {
    stable_tracking_frames_ = 0;
  }

  constexpr std::size_t required_stable_frames = 2;
  if (stable_tracking_frames_ < required_stable_frames) {
    decision.reasons.push_back(RejectReason::Unstable);
  }

  if (current_target_id != -1) {
    last_target_id_ = current_target_id;
  }
  if (current_armor_id != -1) {
    last_armor_id_ = current_armor_id;
  }

  if (input.plan.valid && command_valid) {
    last_command_yaw_ = command_yaw;
    last_command_pitch_ = command_pitch;
    last_fly_time_ = input.plan.fly_time;
  }

  decision.fire_feasible = decision.reasons.empty();

  if (!config_.shoot_enable) {
    decision.reasons.push_back(RejectReason::ShootDisabled);
  }

  decision.shoot = decision.fire_feasible && config_.shoot_enable;
  return decision;
}

void FireEvaluator::reset() noexcept
{
  last_target_id_ = -1;
  last_armor_id_ = -1;
  last_command_yaw_.reset();
  last_command_pitch_.reset();
  last_fly_time_.reset();
  stable_tracking_frames_ = 0;
  last_switch_time_ = {};
}

bool evaluateFire(const L4Planning::AimPlan& plan)
{
  return shouldFire(plan);
}

bool shouldFire(const L4Planning::AimPlan& plan)
{
  return plan.valid && plan.fire_permitted;
}

}  // namespace L5Control
