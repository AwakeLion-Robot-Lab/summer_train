#include "l4_control/fire_decision.hpp"

#include "l5_telemetry/math.hpp"
#include "l2_estimation/armor/vehicle_model.hpp"

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

  L3Estimation::EskfTarget predicted = *input.target;
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
    const Eigen::Vector2d angles{plan.samples.front().yaw, plan.samples.front().pitch};
    return angles.allFinite() ? std::optional{angles} : std::nullopt;
  }
  const Eigen::Vector2d angles{plan.yaw, plan.pitch};
  return angles.allFinite() ? std::optional{angles} : std::nullopt;
}

// 仅供诊断：根据实际回传角计算弹道与运动实体板的交点。
// 开火误差由当前瞄准指令决定，这里的板心坐标不参与放行。
bool actualHit(const FireInput& input, const FireConfig& config, FireDecision& decision)
{
  const auto& plan = input.plan;
  if (!plan.valid || !input.target || plan.target_id != static_cast<int>(input.target->name)
      || !plan.armor_geometry_valid || plan.armor_id < 0
      || plan.armor_id >= input.target->armor_num()
      || !std::isfinite(plan.bullet_speed) || plan.bullet_speed <= 0.0
      || !std::isfinite(plan.fly_time) || plan.fly_time <= 0.0
      || !std::isfinite(plan.gravity) || plan.gravity <= 0.0
      || !std::isfinite(plan.drag_coefficient) || plan.drag_coefficient < 0.0
      || !plan.armor_center_barrel.allFinite() || !plan.armor_center_world.allFinite()
      || !std::isfinite(plan.hit_half_width) || plan.hit_half_width <= 0.0
      || !std::isfinite(plan.hit_half_height) || plan.hit_half_height <= 0.0) {
    return false;
  }
  const Eigen::Vector3d translation = plan.armor_center_barrel - plan.armor_center_world;
  const Eigen::Vector3d initial_velocity = plan.bullet_speed * Eigen::Vector3d{
    std::cos(input.actual_pitch) * std::cos(input.actual_yaw),
    std::cos(input.actual_pitch) * std::sin(input.actual_yaw),
    std::sin(input.actual_pitch)};
  const auto evaluate = [&](double time, Eigen::Vector3d& relative,
                            Eigen::Isometry3d& pose) {
    const auto impact_time = plan.fire_time + std::chrono::duration_cast<L4Planning::TimePoint::duration>(
      std::chrono::duration<double>(time));
    if (impact_time < input.target->t()) return false;
    auto predicted = *input.target;
    predicted.predict(impact_time);
    pose = L3Estimation::VehicleModel::armorPose<double>(
      predicted.rawState().data(), plan.armor_id, predicted.armor_num(), predicted.name);
    double travel = time;
    double drop = 0.5 * plan.gravity * time * time;
    if (plan.drag_coefficient > 1e-8) {
      const double k = plan.drag_coefficient;
      travel = -std::expm1(-k * time) / k;
      drop = plan.gravity * (time - travel) / k;
    }
    const Eigen::Vector3d bullet = initial_velocity * travel - Eigen::Vector3d{0.0, 0.0, drop};
    relative = bullet - translation - pose.translation();
    return relative.allFinite() && pose.matrix().allFinite();
  };

  double time = plan.fly_time;
  Eigen::Vector3d relative;
  Eigen::Isometry3d pose;
  bool converged = false;
  for (int iteration = 0; iteration < 16; ++iteration) {
    if (!evaluate(time, relative, pose)) return false;
    const double distance = pose.linear().col(0).dot(relative);
    if (std::abs(distance) < 1e-5) {
      converged = true;
      break;
    }
    Eigen::Vector3d next_relative;
    Eigen::Isometry3d next_pose;
    constexpr double dt = 1e-4;
    if (!evaluate(time + dt, next_relative, next_pose)) return false;
    const double derivative = (next_pose.linear().col(0).dot(next_relative) - distance) / dt;
    if (!std::isfinite(derivative) || std::abs(derivative) < 1e-3) return false;
    time -= std::clamp(distance / derivative, -0.03, 0.03);
    if (time <= 0.0 || time > 2.0 * plan.fly_time + 0.1) return false;
  }
  if (!converged) return false;
  Eigen::Vector3d after_relative;
  Eigen::Isometry3d after_pose;
  if (!evaluate(time + 1e-4, after_relative, after_pose)
      || (after_pose.linear().col(0).dot(after_relative)
          - pose.linear().col(0).dot(relative)) / 1e-4 <= 1e-3) {
    return false; // 诊断只认可从装甲板可见面进入的交点。
  }
  decision.predicted_fly_time = time;
  decision.predicted_hit_uv = (pose.linear().transpose() * relative).tail<2>();
  // FireConfig 也可能更保守，取两套有效区的交集，始终围绕实体板心。
  const auto type = L3Estimation::armorTypeOf(input.target->name);
  const double width = type == L3Estimation::ArmorType::Big
    ? config.armor_width_big : config.armor_width_small;
  const double half_width = std::min(plan.hit_half_width, 0.5 * width * config.hit_margin_ratio);
  const double half_height = std::min(plan.hit_half_height, 0.5 * config.armor_height * config.hit_margin_ratio);
  return decision.predicted_hit_uv.allFinite()
    && std::abs(decision.predicted_hit_uv.x()) <= half_width
    && std::abs(decision.predicted_hit_uv.y()) <= half_height;
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
  if (!input.bullet_speed_valid) {
    reject(RejectReason::BadBulletSpeed);
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
        if (config_.stop_fire_on_temp_lost) {
          reject(RejectReason::TempLost);
        }
        break;
      case L3Estimation::TrackState::Tracking:
        break;
    }
  }

  const auto command_angles = commandAngles(plan);
  if (!command_angles) {
    reject(RejectReason::PlanInvalid);
  }
  // TempLost 仍可能有有效预测；只有已锁稳且仍在射击窗口内时，才忽略
  // 新观测缺失造成的 tracked_ready=false，避免通过另一项门控再次断火。
  const bool continue_temp_lost = !config_.stop_fire_on_temp_lost && input.target
    && input.track_state == L3Estimation::TrackState::TempLost && plan.valid
    && plan.tracked_phase == L4Planning::ArmorTrackingPhase::Tracking
    && plan.within_firing_window && plan.mpc_aim_feasible;
  if (!plan.mpc_aim_feasible) {
    reject(RejectReason::MpcAimOutsidePlate);
  }
  if (!plan.fire_permitted && !continue_temp_lost && plan.mpc_aim_feasible) {
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

  decision.aim_error_within_tolerance = command_angles && decision.tolerance.valid
    && decision.yaw_error <= decision.tolerance.yaw
    && decision.pitch_error <= decision.tolerance.pitch;
  if (config_.stop_fire_on_aim_error && !decision.aim_error_within_tolerance) {
    reject(RejectReason::AimError);
  }
  decision.hit_inside_plate = actualHit(input, config_, decision);

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
  const double effective_half_width = plan.armor_geometry_valid
    ? std::min(plan.hit_half_width, 0.5 * width * config_.hit_margin_ratio)
    : 0.5 * width * config_.hit_margin_ratio;
  // 容差关于瞄准点对称，不按相对板心的偏移扣减一侧余量。
  const double half_width = effective_half_width * facing;

  const double line_of_sight_pitch = std::atan2(point.z(), horizontal);
  const double tilt = std::abs(std::cos(
    L3Estimation::armorPitchOf(name) + line_of_sight_pitch));
  const double effective_half_height = plan.armor_geometry_valid
    ? std::min(plan.hit_half_height, 0.5 * config_.armor_height * config_.hit_margin_ratio)
    : 0.5 * config_.armor_height * config_.hit_margin_ratio;
  const double half_height = effective_half_height * tilt;

  result.yaw = std::max(std::atan2(half_width, horizontal), config_.min_yaw_tolerance);
  result.pitch = std::max(std::atan2(half_height, slant), config_.min_pitch_tolerance);
  result.valid = std::isfinite(result.yaw) && result.yaw >= 0.0
    && std::isfinite(result.pitch) && result.pitch >= 0.0;
  return result;
}

}  // namespace L5Control
