#include "l5_control/fire_decision.hpp"

#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace L5Control {

FireDecider::FireDecider(FireConfig config) noexcept
: config_(std::move(config))
{
}

AimTolerance FireDecider::tolerance(
  const L4Planning::Plan& plan, L3Estimation::ArmorType type) const noexcept
{
  AimTolerance result;
  if (!plan.fire.has_value() || plan.fire->armor_id < 0) {
    return result;
  }

  const Eigen::Vector3d point = plan.fire->point();
  const double horizontal = std::hypot(point.x(), point.y());
  const double slant = std::hypot(horizontal, point.z());
  if (!std::isfinite(horizontal) || horizontal < 1e-3 || !std::isfinite(slant)) {
    return result;
  }

  const double width = type == L3Estimation::ArmorType::Big
                         ? config_.armor_width_big
                         : config_.armor_width_small;

  // 板面斜对枪口时，水平可命中宽度按 cos(facing_angle) 收缩。
  // 正对时取完整宽度，接近侧对时逐渐收紧到最小 yaw 容差。
  const double facing = std::abs(std::cos(plan.fire->facingAngle()));
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

  // 不短路：一次记录本帧所有拒绝原因，便于回放直接定位多个同时存在的问题。
  if (!config_.shoot_enable) {
    reject(RejectReason::ShootDisabled);
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

  if (!plan.valid()) {
    reject(RejectReason::PlanInvalid);
  }
  if (plan.reason == L4Planning::PlanError::BallisticFailed) {
    reject(RejectReason::BallisticInvalid);
  }
  if (plan.reason == L4Planning::PlanError::BadBulletSpeed) {
    reject(RejectReason::BadBulletSpeed);
  }
  // 命中时刻没有板落在可击发窗口内。高速小陀螺时这是常态间歇，不是故障——
  // 云台照常跟随，只是不开火。
  if (plan.reason == L4Planning::PlanError::OutOfWindow) {
    reject(RejectReason::OutsideHitWindow);
  }
  // TrackOnly 必须有一个可解释的降级原因；否则状态与原因自相矛盾，按无效计划
  // 安全拒绝，避免没有任何拒绝项时 fire_feasible 被误判为 true。
  if (plan.status == L4Planning::PlanStatus::TrackOnly &&
      plan.reason != L4Planning::PlanError::BadBulletSpeed &&
      plan.reason != L4Planning::PlanError::OutOfWindow) {
    reject(RejectReason::PlanInvalid);
  }

  if (!std::isfinite(input.actual_yaw) || !std::isfinite(input.actual_pitch) ||
      !std::isfinite(plan.aim.yaw) || !std::isfinite(plan.aim.pitch)) {
    // 无法计算实际瞄准误差时，本帧必须关火；前面已经收集的原因仍然保留。
    reject(RejectReason::NonFinite);
    decision.shoot = false;
    return decision;
  }

  // 命中判据：实际枪管指向与规划角之差必须落在实体板的角度投影内。
  const auto armor_type =
    input.target.has_value() ? L3Estimation::armorTypeOf(input.target->name)
                             : std::optional<L3Estimation::ArmorType>{};
  decision.tolerance =
    tolerance(plan, armor_type.value_or(L3Estimation::ArmorType::Small));
  decision.yaw_error =
    std::abs(L6Telemetry::limit_rad(plan.aim.yaw - input.actual_yaw));
  decision.pitch_error =
    std::abs(L6Telemetry::limit_rad(plan.aim.pitch - input.actual_pitch));

  if (!decision.tolerance.valid) {
    // 没有实体装甲板可判——中心档下这意味着这一帧本来就不该开火。
    reject(RejectReason::AimError);
  } else if (
    decision.yaw_error > decision.tolerance.yaw ||
    decision.pitch_error > decision.tolerance.pitch) {
    reject(RejectReason::AimError);
  }

  // ShootDisabled 只控制最终输出，不改变理论开火窗口；因此关闭总开关时仍能
  // 通过 fire_feasible 观察判定时序。
  const bool only_disabled = std::all_of(
    decision.reasons.begin(), decision.reasons.end(),
    [](RejectReason reason) { return reason == RejectReason::ShootDisabled; });

  decision.fire_feasible = only_disabled;
  decision.shoot = decision.fire_feasible && config_.shoot_enable;
  return decision;
}

}  // namespace L5Control
