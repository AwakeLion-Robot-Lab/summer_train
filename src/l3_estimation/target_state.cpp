#include "l3_estimation/target_state.hpp"

#include "l6_telemetry/math.hpp"

#include <cmath>
#include <numbers>

namespace L3Estimation {

TargetCovariance targetTransition(double dt) noexcept
{
  TargetCovariance transition = TargetCovariance::Identity();
  transition(CenterX, VelocityX) = dt;
  transition(CenterY, VelocityY) = dt;
  transition(CenterZ, VelocityZ) = dt;
  transition(Yaw, Vyaw) = dt;
  return transition;
}

TargetCovariance targetProcessNoise(
  double dt,
  bool outpost,
  const TargetConfig& config) noexcept
{
  const double translation_variance =
    outpost ? config.outpost_accel_variance : config.accel_variance;
  const double yaw_variance =
    outpost ? config.outpost_yaw_accel_variance : config.yaw_accel_variance;

  const double dt2 = dt * dt;
  Eigen::Matrix2d unit_noise;
  unit_noise << dt2 * dt2 / 4.0, dt2 * dt / 2.0,
                dt2 * dt / 2.0, dt2;

  TargetCovariance noise = TargetCovariance::Zero();
  noise.block<2, 2>(CenterX, CenterX) = translation_variance * unit_noise;
  noise.block<2, 2>(CenterY, CenterY) = translation_variance * unit_noise;
  noise.block<2, 2>(CenterZ, CenterZ) = translation_variance * unit_noise;
  noise.block<2, 2>(Yaw, Yaw) = yaw_variance * unit_noise;
  return noise;
}

TargetStateVector predictTargetState(
  const TargetStateVector& state,
  double dt) noexcept
{
  TargetStateVector predicted = targetTransition(dt) * state;
  predicted[Yaw] = L6Telemetry::limit_rad(predicted[Yaw]);
  return predicted;
}

Eigen::Vector3d armorPosition(
  const TargetStateVector& state,
  int armor_count,
  int armor_id) noexcept
{
  if (armor_count <= 0) {
    return Eigen::Vector3d::Zero();
  }

  const double angle = L6Telemetry::limit_rad(
    state[Yaw] + static_cast<double>(armor_id) * 2.0 * std::numbers::pi /
                   static_cast<double>(armor_count));
  const bool alternate = armor_count == 4 && (armor_id % 2 != 0);
  const double radius =
    alternate ? state[RadiusA] + state[RadiusDifference] : state[RadiusA];
  return {
    state[CenterX] - radius * std::cos(angle),
    state[CenterY] - radius * std::sin(angle),
    alternate ? state[CenterZ] + state[HeightDifference] : state[CenterZ]};
}

std::vector<Eigen::Vector4d> armorPoses(
  const TargetStateVector& state,
  int armor_count)
{
  std::vector<Eigen::Vector4d> armors;
  if (armor_count <= 0) {
    return armors;
  }

  armors.reserve(static_cast<std::size_t>(armor_count));
  for (int id = 0; id < armor_count; ++id) {
    const double angle = L6Telemetry::limit_rad(
      state[Yaw] + static_cast<double>(id) * 2.0 * std::numbers::pi /
                     static_cast<double>(armor_count));
    const Eigen::Vector3d position = armorPosition(state, armor_count, id);
    armors.emplace_back(position.x(), position.y(), position.z(), angle);
  }
  return armors;
}

}  // namespace L3Estimation
