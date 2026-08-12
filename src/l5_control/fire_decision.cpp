#include "l5_control/fire_decision.hpp"

#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>

namespace L5Control {

AimTolerance FireDecider::tolerance(
  const L4Planning::Plan& plan, L3Estimation::ArmorType type) const noexcept
{
  AimTolerance result;
  if (plan.fire_armor_id < 0) {
    return result;
  }

  const auto& point = plan.fire_armor_point;
  const double horizontal = std::hypot(point.x(), point.y());
  const double slant = std::hypot(horizontal, point.z());
  if (!std::isfinite(horizontal) || horizontal < 1e-3 || !std::isfinite(slant)) {
    return result;
  }

  const double width = type == L3Estimation::ArmorType::Big
                         ? config_.armor_width_big
                         : config_.armor_width_small;

  // 板面斜对枪口时，投影到视线方向的宽度按 cos 收缩——这一项是 jlu 和
  // rm.cv.fans 都有而 talos / FYT 都没有的。缺了它，转到 60° 的板会被当成
  // 正对的板给出同样宽的容差，等于在最容易脱靶的姿态下最宽容。
  const double facing = std::abs(std::cos(plan.fire_delta_angle));
  const double half_width = 0.5 * width * config_.hit_margin_ratio * facing;
  const double half_height = 0.5 * config_.armor_height * config_.hit_margin_ratio;

  // yaw 是水平角，用水平距离；pitch 是竖直角，用斜距。
  result.yaw = std::max(std::atan2(half_width, horizontal), config_.min_yaw_tolerance);
  result.pitch = std::max(std::atan2(half_height, slant), config_.min_pitch_tolerance);
  result.valid = std::isfinite(result.yaw) && std::isfinite(result.pitch);
  return result;
}

FireDecision FireDecider::decide(const FireInput& input) const
{
  FireDecision decision;
  const auto& plan = input.plan;

  const auto reject = [&decision](RejectReason reason) {
    decision.reasons.push_back(reason);
  };

  // 不短路：把所有不满足的条件都记下来。只报第一条的话，实车上"为什么不开火"
  // 这个问题要靠反复复现来一条条剥，而每复现一次就是一次上场。
  if (!config_.shoot_enable) {
    reject(RejectReason::ShootDisabled);
  }
  if (!config_.parametersReady()) {
    reject(RejectReason::MissingCalibration);
  }
  if (!input.calibration_ready) {
    reject(RejectReason::MissingCalibration);
  }
  if (!input.serial_fresh) {
    reject(RejectReason::RobotStateStale);
  }
  if (!input.gimbal_pose_fresh) {
    reject(RejectReason::GimbalPoseStale);
  }
  if (input.armor_switching) {
    reject(RejectReason::ArmorSwitching);
  }
  if (input.command_jump) {
    reject(RejectReason::CommandJump);
  }

  if (!input.target.has_value()) {
    reject(RejectReason::NoTarget);
  } else {
    switch (input.track_state) {
      case L3Estimation::TrackState::Lost:
      case L3Estimation::TrackState::Detecting:
        reject(RejectReason::NotTracking);
        break;
      case L3Estimation::TrackState::TempLost:
        // 短时丢失时状态全靠外推，位置误差随丢失时长增长，不允许开火。
        reject(RejectReason::TempLost);
        break;
      case L3Estimation::TrackState::Tracking:
        break;
    }
  }

  if (!plan.valid) {
    reject(RejectReason::PlanInvalid);
  }
  if (!plan.ballistic_valid) {
    reject(RejectReason::BallisticInvalid);
  }
  if (plan.error == L4Planning::PlanError::BadBulletSpeed) {
    reject(RejectReason::BadBulletSpeed);
  }
  // 命中时刻没有板落在可击发窗口内。高速小陀螺时这是常态间歇，不是故障——
  // 云台照常跟随，只是不开火。
  if (!plan.fire_admissible) {
    reject(RejectReason::OutsideHitWindow);
  }

  if (config_.heat_limit.has_value() &&
      input.robot_state.heat >= *config_.heat_limit) {
    reject(RejectReason::HeatLimit);
  }

  if (!std::isfinite(input.actual_yaw) || !std::isfinite(input.actual_pitch) ||
      !std::isfinite(plan.yaw) || !std::isfinite(plan.pitch)) {
    reject(RejectReason::NonFinite);
    decision.shoot = false;
    return decision;
  }

  // 云台限位。未标定时 parametersReady() 已经拦下了，这里只在有值时判。
  const bool yaw_in_range =
    !config_.min_yaw.has_value() || !config_.max_yaw.has_value() ||
    (plan.yaw >= *config_.min_yaw && plan.yaw <= *config_.max_yaw);
  const bool pitch_in_range =
    !config_.min_pitch.has_value() || !config_.max_pitch.has_value() ||
    (plan.pitch >= *config_.min_pitch && plan.pitch <= *config_.max_pitch);
  if (!yaw_in_range || !pitch_in_range) {
    reject(RejectReason::OutOfRange);
  }

  // 命中判据：实际指向与规划指向之差，对上装甲板在该距离上张开的角度。
  const auto armor_type =
    input.target.has_value() ? L3Estimation::armorTypeOf(input.target->name)
                             : std::optional<L3Estimation::ArmorType>{};
  decision.tolerance =
    tolerance(plan, armor_type.value_or(L3Estimation::ArmorType::Small));
  decision.yaw_error = std::abs(L6Telemetry::limit_rad(plan.yaw - input.actual_yaw));
  decision.pitch_error =
    std::abs(L6Telemetry::limit_rad(plan.pitch - input.actual_pitch));

  if (!decision.tolerance.valid) {
    // 没有实体装甲板可判——中心档下这意味着这一帧本来就不该开火。
    reject(RejectReason::AimError);
  } else if (
    decision.yaw_error > decision.tolerance.yaw ||
    decision.pitch_error > decision.tolerance.pitch) {
    reject(RejectReason::AimError);
  }

  // fire_feasible 是"理论上这一枪该打"，只排除 ShootDisabled 这一条人为闸门；
  // shoot 才是真正下发的值。两者分开，验收阶段可以在 shoot_enable = false 下
  // 观察 fire_feasible 的时序而不真的打出去。
  const bool only_disabled = std::all_of(
    decision.reasons.begin(), decision.reasons.end(),
    [](RejectReason reason) { return reason == RejectReason::ShootDisabled; });

  decision.fire_feasible = only_disabled;
  decision.shoot = decision.fire_feasible && config_.shoot_enable;
  return decision;
}

bool shouldFire(const L4Planning::AimPlan& plan)
{
  return plan.valid && plan.fire_admissible;
}

}  // namespace L5Control
