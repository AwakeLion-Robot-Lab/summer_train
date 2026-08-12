#include "l3_estimation/tracked_target.hpp"

#include "l6_telemetry/math.hpp"

#include <cmath>

namespace L3Estimation {

TrackedTarget::TrackedTarget(
  ArmorName target_name,
  ArmorType target_armor_type,
  int armor_count,
  TimePoint timestamp,
  const TargetStateVector& state,
  const TargetCovariance& covariance,
  TargetConfig config)
: name(target_name),
  armor_type(target_armor_type),
  config_(config),
  armor_count_(armor_count),
  timestamp_(timestamp),
  state_(state),
  covariance_(covariance)
{
}

TrackedTarget::TrackedTarget(
  double x,
  double vyaw,
  double radius,
  double height,
  double yaw)
: armor_count_(4)
{
  state_ << x, 0.0, 0.0, 0.0, 0.0, 0.0,
    L6Telemetry::limit_rad(yaw), vyaw, radius, 0.0, height;
}

bool TrackedTarget::valid() const noexcept
{
  return armor_count_ > 0 && state_.allFinite() && covariance_.allFinite();
}

void TrackedTarget::predict(TimePoint timestamp)
{
  predict(L6Telemetry::delta_time(timestamp, timestamp_));
  timestamp_ = timestamp;
}

void TrackedTarget::predict(double dt)
{
  const TargetCovariance transition = targetTransition(dt);
  covariance_ = transition * covariance_ * transition.transpose() +
                targetProcessNoise(dt, name == ArmorName::Outpost, config_);
  state_ = predictTargetState(state_, dt);
}

std::vector<Eigen::Vector4d> TrackedTarget::armorPoses() const
{
  return L3Estimation::armorPoses(state_, armor_count_);
}

}  // namespace L3Estimation
