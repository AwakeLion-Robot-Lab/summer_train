#include "l3_estimation/filter_est/target.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace L3Estimation::FilterEst {
namespace {

Eigen::VectorXd addState(const Eigen::VectorXd& state, const Eigen::VectorXd& delta)
{
  Eigen::VectorXd result = state + delta;
  result[Yaw] = L6Telemetry::limit_rad(result[Yaw]);
  return result;
}

TargetStateVector fixedState(const Eigen::VectorXd& state)
{
  TargetStateVector result = TargetStateVector::Zero();
  if (state.size() == kTargetStateSize) {
    result = state;
  }
  return result;
}

TargetCovariance fixedCovariance(const Eigen::MatrixXd& covariance)
{
  TargetCovariance result = TargetCovariance::Zero();
  if (covariance.rows() == kTargetStateSize && covariance.cols() == kTargetStateSize) {
    result = covariance;
  }
  return result;
}

}  // namespace

Target::Target(
  const Armor& armor,
  TimePoint timestamp,
  double radius,
  int armor_count,
  const Eigen::VectorXd& initial_covariance_diagonal,
  L3Estimation::TargetConfig target_config,
  TargetConfig filter_config)
: target_config_(target_config),
  filter_config_(filter_config),
  name_(armor.name),
  armor_type_(armor.type),
  armor_count_(armor_count),
  timestamp_(timestamp)
{
  TargetStateVector initial = TargetStateVector::Zero();
  const Eigen::Vector3d& position = armor.xyz_in_world;
  const double armor_yaw = armor.ypr_in_world[0];
  initial << position.x() + radius * std::cos(armor_yaw), 0.0,
    position.y() + radius * std::sin(armor_yaw), 0.0,
    position.z(), 0.0, armor_yaw, 0.0, radius, 0.0, 0.0;

  TargetCovariance covariance = TargetCovariance::Identity();
  if (initial_covariance_diagonal.size() == kTargetStateSize) {
    covariance = initial_covariance_diagonal.asDiagonal();
  }
  filter_ = ExtendedKalmanFilter(initial, covariance, addState);
}

void Target::predict(TimePoint timestamp)
{
  const double dt = L6Telemetry::delta_time(timestamp, timestamp_);
  timestamp_ = timestamp;

  if (name_ == ArmorName::Outpost && converged() && std::abs(filter_.x[Vyaw]) > 2.0) {
    filter_.x[Vyaw] = filter_.x[Vyaw] > 0.0 ? filter_config_.outpost_v_yaw
                                            : -filter_config_.outpost_v_yaw;
  }

  const TargetCovariance transition = targetTransition(dt);
  const TargetCovariance noise =
    targetProcessNoise(dt, name_ == ArmorName::Outpost, target_config_);
  filter_.predict(transition, noise, [dt](const Eigen::VectorXd& state) {
    return predictTargetState(fixedState(state), dt);
  });
}

void Target::update(const Armor& armor)
{
  const std::vector<Eigen::Vector4d> predicted = armorPoses();
  std::vector<std::pair<Eigen::Vector4d, int>> candidates;
  candidates.reserve(predicted.size());
  for (int id = 0; id < armor_count_; ++id) {
    candidates.emplace_back(predicted[static_cast<std::size_t>(id)], id);
  }
  std::sort(
    candidates.begin(), candidates.end(),
    [](const std::pair<Eigen::Vector4d, int>& lhs,
       const std::pair<Eigen::Vector4d, int>& rhs) {
      return L6Telemetry::xyz2ypd(lhs.first.head<3>()).z() <
             L6Telemetry::xyz2ypd(rhs.first.head<3>()).z();
    });

  int matched_id = -1;
  double minimum_angle_error = std::numeric_limits<double>::infinity();
  const int candidate_count = std::min(3, armor_count_);
  for (int index = 0; index < candidate_count; ++index) {
    const Eigen::Vector4d& candidate = candidates[static_cast<std::size_t>(index)].first;
    const Eigen::Vector3d candidate_ypd =
      L6Telemetry::xyz2ypd(candidate.head<3>());
    const double angle_error =
      std::abs(L6Telemetry::limit_rad(armor.ypr_in_world[0] - candidate[3])) +
      std::abs(L6Telemetry::limit_rad(armor.ypd_in_world.x() - candidate_ypd.x()));
    if (angle_error < minimum_angle_error) {
      minimum_angle_error = angle_error;
      matched_id = candidates[static_cast<std::size_t>(index)].second;
    }
  }

  if (matched_id < 0) {
    return;
  }
  jumped_ = jumped_ || matched_id != 0;
  last_id_ = matched_id;
  ++update_count_;
  updateObservation(armor, matched_id);
}

void Target::updateObservation(const Armor& armor, int armor_id)
{
  const double center_yaw =
    std::atan2(armor.xyz_in_world.y(), armor.xyz_in_world.x());
  const double delta_angle =
    L6Telemetry::limit_rad(armor.ypr_in_world[0] - center_yaw);

  Eigen::Vector4d noise_diagonal;
  noise_diagonal << filter_config_.angle_variance, filter_config_.angle_variance,
    filter_config_.distance_variance + std::log1p(std::abs(delta_angle)),
    filter_config_.armor_yaw_variance +
      std::log1p(std::abs(armor.ypd_in_world.z())) / 200.0;

  const auto observation = [this, armor_id](const Eigen::VectorXd& state) {
    const TargetStateVector fixed = fixedState(state);
    const Eigen::Vector3d ypd =
      L6Telemetry::xyz2ypd(armorPosition(fixed, armor_count_, armor_id));
    const double angle = L6Telemetry::limit_rad(
      fixed[Yaw] + static_cast<double>(armor_id) * 2.0 * std::numbers::pi /
                     static_cast<double>(armor_count_));
    return Eigen::Vector4d{ypd.x(), ypd.y(), ypd.z(), angle};
  };
  const auto subtract = [](const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs) {
    Eigen::VectorXd result = lhs - rhs;
    result[0] = L6Telemetry::limit_rad(result[0]);
    result[1] = L6Telemetry::limit_rad(result[1]);
    result[3] = L6Telemetry::limit_rad(result[3]);
    return result;
  };

  Eigen::Vector4d measured;
  measured << armor.ypd_in_world.x(), armor.ypd_in_world.y(),
    armor.ypd_in_world.z(), armor.ypr_in_world[0];
  filter_.update(
    measured,
    observationJacobian(armor_id),
    noise_diagonal.asDiagonal(),
    observation,
    subtract);
}

Eigen::MatrixXd Target::observationJacobian(int armor_id) const
{
  const TargetStateVector state = fixedState(filter_.x);
  const double angle = L6Telemetry::limit_rad(
    state[Yaw] + static_cast<double>(armor_id) * 2.0 * std::numbers::pi /
                   static_cast<double>(armor_count_));
  const bool alternate = armor_count_ == 4 && armor_id % 2 != 0;
  const double radius =
    alternate ? state[RadiusA] + state[RadiusDifference] : state[RadiusA];

  const double dx_dyaw = radius * std::sin(angle);
  const double dy_dyaw = -radius * std::cos(angle);
  const double dx_dr = -std::cos(angle);
  const double dy_dr = -std::sin(angle);
  const double dx_ddr = alternate ? dx_dr : 0.0;
  const double dy_ddr = alternate ? dy_dr : 0.0;
  const double dz_ddz = alternate ? 1.0 : 0.0;

  Eigen::Matrix<double, 4, kTargetStateSize> pose_jacobian;
  pose_jacobian <<
    1, 0, 0, 0, 0, 0, dx_dyaw, 0, dx_dr, dx_ddr, 0,
    0, 0, 1, 0, 0, 0, dy_dyaw, 0, dy_dr, dy_ddr, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dz_ddz,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;

  Eigen::Matrix4d spherical_jacobian = Eigen::Matrix4d::Zero();
  spherical_jacobian.topLeftCorner<3, 3>() = L6Telemetry::xyz2ypdJacobian(
    armorPosition(state, armor_count_, armor_id));
  spherical_jacobian(3, 3) = 1.0;
  return spherical_jacobian * pose_jacobian;
}

std::vector<Eigen::Vector4d> Target::armorPoses() const
{
  return L3Estimation::armorPoses(fixedState(filter_.x), armor_count_);
}

TrackedTarget Target::snapshot() const
{
  TrackedTarget result(
    name_, armor_type_, armor_count_, timestamp_, fixedState(filter_.x),
    fixedCovariance(filter_.P), target_config_);
  result.jumped = jumped_;
  result.last_id = last_id_;
  result.normalized_innovation_squared = filter_.last_nis;
  return result;
}

bool Target::diverged() const
{
  const TargetStateVector state = fixedState(filter_.x);
  const double radius_a = state[RadiusA];
  const double radius_b = radius_a + state[RadiusDifference];
  const auto in_range = [this](double radius) {
    return radius > target_config_.min_radius && radius < target_config_.max_radius;
  };
  if (filter_.x.size() == kTargetStateSize && filter_.x.allFinite() &&
      filter_.P.allFinite() && in_range(radius_a) && in_range(radius_b)) {
    return false;
  }
  L6Telemetry::logDebug("FilterEst target diverged: r1, r2", radius_a, radius_b);
  return true;
}

bool Target::converged()
{
  const int required = name_ == ArmorName::Outpost
                         ? filter_config_.outpost_min_update_count
                         : filter_config_.min_update_count;
  if (update_count_ > required && !diverged()) {
    converged_ = true;
  }
  return converged_;
}

}  // namespace L3Estimation::FilterEst
