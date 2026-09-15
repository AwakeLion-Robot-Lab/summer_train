#include "l5_control/fire_decision.hpp"

#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace L5Control {
namespace {

std::optional<Eigen::Vector4d> selectedArmorPose(const FireInput& input)
{
  if (!input.target || input.plan.armor_id < 0 ||
      input.plan.impact_time < input.target->t()) {
    return std::nullopt;
  }

  L3Estimation::TrackedTarget predicted = *input.target;
  predicted.predict(input.plan.impact_time);
  const auto armors = predicted.armor_xyza_list();
  const auto index = static_cast<std::size_t>(input.plan.armor_id);
  if (index >= armors.size() || !armors[index].allFinite()) {
    return std::nullopt;
  }
  return armors[index];
}

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

FireDecider::FireDecider(FireConfig config) noexcept
: config_(std::move(config))
{
}

FireDecision FireDecider::decide(const FireInput& input) const
{
  FireDecision decision;
  const auto& plan = input.plan;
  const auto reject = [&decision](RejectReason reason) {
    decision.reasons.push_back(reason);
  };

  if (!config_.shoot_enable) {
    reject(RejectReason::ShootDisabled);
  }
  if (input.command_jump) {
    reject(RejectReason::CommandJump);
  }

  if (!input.target) {
    reject(RejectReason::NoTarget);
  } else {
    switch (input.track_state) {
      case L3Estimation::TrackState::Lost:
      case L3Estimation::TrackState::Detecting:
        reject(RejectReason::NotTracking);
        break;
      case L3Estimation::TrackState::TempLost:
        reject(RejectReason::TempLost);
        break;
      case L3Estimation::TrackState::Tracking:
        break;
    }
  }

  const auto command_angles = commandAngles(plan);
  if (!command_angles) {
    reject(RejectReason::PlanInvalid);
  }
  if (!plan.fire_permitted) {
    reject(RejectReason::OutsideHitWindow);
  }

  if (!std::isfinite(input.actual_yaw) || !std::isfinite(input.actual_pitch)) {
    reject(RejectReason::NonFinite);
    return decision;
  }

  const auto armor_type = input.target
    ? L3Estimation::armorTypeOf(input.target->name)
    : std::optional<L3Estimation::ArmorType>{};
  const auto armor_name = input.target
    ? input.target->name
    : L3Estimation::ArmorName::Unknown;
  decision.tolerance = tolerance(
    plan,
    selectedArmorPose(input),
    armor_type.value_or(L3Estimation::ArmorType::Small),
    armor_name);

  if (command_angles) {
    decision.yaw_error = std::abs(
      L6Telemetry::limit_rad((*command_angles)[0] - input.actual_yaw));
    decision.pitch_error = std::abs(
      L6Telemetry::limit_rad((*command_angles)[1] - input.actual_pitch));
  }

  if (!decision.tolerance.valid ||
      decision.yaw_error > decision.tolerance.yaw ||
      decision.pitch_error > decision.tolerance.pitch) {
    reject(RejectReason::AimError);
  }

  const bool only_disabled = std::all_of(
    decision.reasons.begin(), decision.reasons.end(),
    [](RejectReason reason) { return reason == RejectReason::ShootDisabled; });
  decision.fire_feasible = only_disabled;
  decision.shoot = decision.fire_feasible && config_.shoot_enable;
  return decision;
}

AimTolerance FireDecider::tolerance(
  const L4Planning::AimPlan& plan,
  const std::optional<Eigen::Vector4d>& armor_pose,
  L3Estimation::ArmorType type,
  L3Estimation::ArmorName name) const noexcept
{
  AimTolerance result;
  if (!plan.valid || plan.armor_id < 0 || !armor_pose) {
    return result;
  }

  const Eigen::Vector3d point = plan.aim_point_barrel;
  const double horizontal = std::hypot(point.x(), point.y());
  const double slant = point.norm();
  if (!point.allFinite() || horizontal < 1e-3 || !std::isfinite(slant)) {
    return result;
  }

  const double width = type == L3Estimation::ArmorType::Big
    ? config_.armor_width_big
    : config_.armor_width_small;
  const double line_of_sight = std::atan2(point.y(), point.x());
  const double facing = std::abs(std::cos(
    std::remainder(line_of_sight - armor_pose->w(), 2.0 * std::numbers::pi)));
  const double half_width = 0.5 * width * config_.hit_margin_ratio * facing;

  const double line_of_sight_pitch = std::atan2(point.z(), horizontal);
  const double tilt = std::abs(std::cos(
    L3Estimation::armorPitchOf(name) + line_of_sight_pitch));
  const double half_height =
    0.5 * config_.armor_height * config_.hit_margin_ratio * tilt;

  result.yaw = std::max(
    std::atan2(half_width, horizontal), config_.min_yaw_tolerance);
  result.pitch = std::max(
    std::atan2(half_height, slant), config_.min_pitch_tolerance);
  result.valid = std::isfinite(result.yaw) && std::isfinite(result.pitch);
  return result;
}

}  // namespace L5Control
