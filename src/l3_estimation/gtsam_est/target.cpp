#include "l3_estimation/gtsam_est/target.hpp"

#ifdef NEWVISION_USE_GTSAM

#include "l6_telemetry/math.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/PriorFactor.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace L3Estimation::GtsamEst {
namespace {

[[nodiscard]] double seconds(TimePoint current, TimePoint previous)
{
  return std::chrono::duration<double>(current - previous).count();
}

[[nodiscard]] double logistic(double raw, double minimum, double maximum)
{
  const double unit = raw > 0.0 ? 1.0 / (1.0 + std::exp(-raw))
                                : std::exp(raw) / (1.0 + std::exp(raw));
  return minimum + unit * (maximum - minimum);
}

[[nodiscard]] double logisticInverse(double value, double minimum, double maximum)
{
  const double margin = std::max(1e-9, (maximum - minimum) * 1e-6);
  const double bounded = std::clamp(value, minimum + margin, maximum - margin);
  return std::log((bounded - minimum) / (maximum - bounded));
}

[[nodiscard]] double logisticDerivative(
  double output,
  double minimum,
  double maximum)
{
  return (output - minimum) * (maximum - output) / (maximum - minimum);
}

[[nodiscard]] gtsam::Pose3 cameraPose(const Armor& armor)
{
  return gtsam::Pose3{
    gtsam::Rot3{L6Telemetry::yprToRotation(armor.ypr_in_camera)},
    armor.xyz_in_camera};
}

[[nodiscard]] double initialRadius(
  ArmorName name,
  const TargetConfig& target_config,
  const Config& gtsam_config)
{
  if (name == ArmorName::Outpost) {
    return target_config.outpost_radius;
  }
  return gtsam_config.default_radius;
}

[[nodiscard]] int armorCountForName(ArmorName name)
{
  return name == ArmorName::Outpost || name == ArmorName::BaseSmall ||
             name == ArmorName::BaseLarge
           ? 3
           : 4;
}

[[nodiscard]] gtsam::SharedNoiseModel isotropic(int dimension, double sigma)
{
  return gtsam::noiseModel::Isotropic::Sigma(dimension, sigma);
}

}  // namespace

namespace keys {

gtsam::Key center(std::uint64_t k) noexcept { return gtsam::Symbol{'x', k}; }
gtsam::Key velocity(std::uint64_t k) noexcept { return gtsam::Symbol{'v', k}; }
gtsam::Key yaw(std::uint64_t k) noexcept { return gtsam::Symbol{'r', k}; }
gtsam::Key vyaw(std::uint64_t k) noexcept { return gtsam::Symbol{'w', k}; }
gtsam::Key radiusA() noexcept { return gtsam::Symbol{'a', 0}; }
gtsam::Key radiusB() noexcept { return gtsam::Symbol{'b', 0}; }
gtsam::Key deltaZ() noexcept { return gtsam::Symbol{'z', 0}; }

gtsam::Key armorPose(std::uint64_t k, int armor_id)
{
  constexpr std::array<char, 4> prefixes{'h', 'j', 'k', 'l'};
  if (armor_id < 0 || armor_id >= static_cast<int>(prefixes.size())) {
    throw std::out_of_range("invalid physical armor id");
  }
  return gtsam::Symbol{
    static_cast<unsigned char>(prefixes[static_cast<std::size_t>(armor_id)]), k};
}

}  // namespace keys

Target::Target(
  TargetConfig target_config,
  Config gtsam_config,
  ArmorConfig armor_config,
  const L1Sensor::CameraCalibration& calibration)
: target_config_(target_config),
  gtsam_config_(gtsam_config),
  armor_config_(armor_config),
  calibration_(calibration)
{
}

void Target::initialize(
  const Armor& armor,
  TimePoint timestamp,
  const Eigen::Isometry3d& T_world_camera)
{
  name_ = armor.name;
  armor_type_ = armor.type;
  armor_count_ = armorCountForName(name_);
  timestamp_ = timestamp;
  last_observation_timestamp_ = timestamp;
  k_ = 0;
  failed_ = false;
  last_id_ = 0;

  initializeStateFromArmor(armor);
  const double radius = state_[RadiusA];

  covariance_ = TargetCovariance::Zero();
  covariance_.diagonal() <<
    gtsam_config_.translation_prior_sigma * gtsam_config_.translation_prior_sigma,
    gtsam_config_.velocity_prior_sigma * gtsam_config_.velocity_prior_sigma,
    gtsam_config_.translation_prior_sigma * gtsam_config_.translation_prior_sigma,
    gtsam_config_.velocity_prior_sigma * gtsam_config_.velocity_prior_sigma,
    gtsam_config_.translation_prior_sigma * gtsam_config_.translation_prior_sigma,
    gtsam_config_.velocity_prior_sigma * gtsam_config_.velocity_prior_sigma,
    gtsam_config_.yaw_prior_sigma * gtsam_config_.yaw_prior_sigma,
    gtsam_config_.vyaw_prior_sigma * gtsam_config_.vyaw_prior_sigma,
    gtsam_config_.radius_prior_sigma * gtsam_config_.radius_prior_sigma,
    armor_count_ == 4
      ? 2.0 * gtsam_config_.radius_prior_sigma * gtsam_config_.radius_prior_sigma
      : 0.0,
    armor_count_ == 4 ? gtsam_config_.dz_prior_sigma * gtsam_config_.dz_prior_sigma
                      : 0.0;

  addMotionState(0, 0.0);
  addStaticGeometry(radius);
  addArmorFactors(0, armor, 0, T_world_camera);
  // JLU 的 first_update_batch_size 表示第一次提交时最小的 k。配置为 1 时，
  // 先缓存 k=0，等 k=1 的运动约束也进入初始图后再做第一次优化。
  if (k_ >= static_cast<std::uint64_t>(gtsam_config_.first_update_batch_size)) {
    commitPendingGraph();
  }
}

bool Target::update(
  const std::vector<const Armor*>& armors,
  TimePoint timestamp,
  const Eigen::Isometry3d& T_world_camera)
{
  if (failed_ || armor_count_ <= 0) {
    return false;
  }
  if (timestamp <= timestamp_) {
    failed_ = true;
    return false;
  }
  if (timestamp - last_observation_timestamp_ > gtsam_config_.lost_threshold) {
    failed_ = true;
    return false;
  }

  const double dt = seconds(timestamp, timestamp_);
  timestamp_ = timestamp;
  ++k_;
  // JLU 冷启动在 k < first_update_batch_size 时每帧都从当帧第一块板
  // 重新构造整车初值，到 k == batch_size 再联合提交初始图。
  if (k_ < static_cast<std::uint64_t>(gtsam_config_.first_update_batch_size) &&
      !armors.empty() && armors.front() && armors.front()->name == name_ &&
      armors.front()->type == armor_type_) {
    initializeStateFromArmor(*armors.front());
  } else {
    const TargetCovariance transition = targetTransition(dt);
    covariance_ = transition * covariance_ * transition.transpose() +
                  targetProcessNoise(
                    dt, name_ == ArmorName::Outpost, target_config_);
    state_ = predictTargetState(state_, dt);
  }
  addMotionState(k_, dt);

  bool found = false;
  std::vector<bool> used_ids(static_cast<std::size_t>(armor_count_), false);
  for (const Armor* armor : armors) {
    if (!armor || armor->name != name_ || armor->type != armor_type_) {
      continue;
    }
    const std::optional<int> matched = matchArmor(*armor, used_ids);
    if (!matched) {
      continue;
    }
    const int armor_id = *matched;
    used_ids[static_cast<std::size_t>(armor_id)] = true;
    addArmorFactors(k_, *armor, armor_id, T_world_camera);
    found = true;
    last_id_ = armor_id;
  }
  if (found) {
    last_observation_timestamp_ = timestamp;
  }

  if (isam_started_ ||
      k_ >= static_cast<std::uint64_t>(gtsam_config_.first_update_batch_size)) {
    commitPendingGraph();
  }
  return found;
}

void Target::initializeStateFromArmor(const Armor& armor)
{
  const double radius = initialRadius(name_, target_config_, gtsam_config_);
  const double armor_yaw = armor.ypr_raw_in_world[0];
  state_ <<
    armor.xyz_in_world.x() + radius * std::cos(armor_yaw), 0.0,
    armor.xyz_in_world.y() + radius * std::sin(armor_yaw), 0.0,
    armor.xyz_in_world.z(), 0.0, armor_yaw, 0.0, radius, 0.0,
    gtsam_config_.default_dz;
}

std::optional<int> Target::matchArmor(
  const Armor& armor,
  const std::vector<bool>& used_ids) const
{
  const std::vector<Eigen::Vector4d> predicted = armorPoses();
  std::optional<int> best;
  double best_yaw_error = std::numeric_limits<double>::infinity();
  double best_distance = std::numeric_limits<double>::infinity();
  for (int id = 0; id < armor_count_; ++id) {
    if (used_ids[static_cast<std::size_t>(id)]) {
      continue;
    }
    const Eigen::Vector4d& candidate = predicted[static_cast<std::size_t>(id)];
    const double distance =
      (candidate.head<3>() - armor.xyz_in_world).norm();
    const double yaw_error = std::abs(
      L6Telemetry::limit_rad(candidate[3] - armor.ypr_raw_in_world[0]));
    if (distance > gtsam_config_.max_match_distance ||
        yaw_error > gtsam_config_.max_match_yaw_diff) {
      continue;
    }
    if (yaw_error < best_yaw_error ||
        (std::abs(yaw_error - best_yaw_error) < 1e-12 && distance < best_distance)) {
      best = id;
      best_yaw_error = yaw_error;
      best_distance = distance;
    }
  }
  return best;
}

void Target::addMotionState(std::uint64_t k, double dt)
{
  const gtsam::Point3 center_value{
    state_[CenterX], state_[CenterY], state_[CenterZ]};
  const gtsam::Vector3 velocity_value{
    state_[VelocityX], state_[VelocityY], state_[VelocityZ]};
  const gtsam::Rot2 yaw_value = gtsam::Rot2::fromAngle(state_[Yaw]);

  pending_values_.insert(keys::center(k), center_value);
  pending_values_.insert(keys::velocity(k), velocity_value);
  pending_values_.insert(keys::yaw(k), yaw_value);
  pending_values_.insert(keys::vyaw(k), state_[Vyaw]);

  if (k == 0) {
    pending_graph_.addPrior(
      keys::center(0), center_value,
      isotropic(3, gtsam_config_.translation_prior_sigma));
    pending_graph_.addPrior(
      keys::velocity(0), velocity_value,
      isotropic(3, gtsam_config_.velocity_prior_sigma));
    pending_graph_.addPrior(
      keys::yaw(0), yaw_value,
      isotropic(1, gtsam_config_.yaw_prior_sigma));
    pending_graph_.addPrior(
      keys::vyaw(0), state_[Vyaw],
      isotropic(1, gtsam_config_.vyaw_prior_sigma));
    return;
  }

  pending_graph_.emplace_shared<TranslationFactor>(
    isotropic(3, gtsam_config_.translation_factor_sigma),
    keys::center(k - 1), keys::velocity(k - 1), keys::center(k), dt);
  pending_graph_.emplace_shared<VelocityFactor>(
    isotropic(3, gtsam_config_.velocity_factor_sigma),
    keys::velocity(k - 1), keys::velocity(k));
  pending_graph_.emplace_shared<YawFactor>(
    isotropic(1, gtsam_config_.yaw_factor_sigma),
    keys::yaw(k - 1), keys::vyaw(k - 1), keys::yaw(k), dt);
  pending_graph_.emplace_shared<VyawFactor>(
    isotropic(1, gtsam_config_.vyaw_factor_sigma),
    keys::vyaw(k - 1), keys::vyaw(k));
}

void Target::addStaticGeometry(double initial_radius)
{
  const double raw_radius = logisticInverse(
    initial_radius, gtsam_config_.radius_min, gtsam_config_.radius_max);
  pending_values_.insert(keys::radiusA(), raw_radius);
  pending_graph_.addPrior(
    keys::radiusA(), raw_radius,
    isotropic(1, gtsam_config_.radius_prior_sigma));

  if (armor_count_ == 4) {
    pending_values_.insert(keys::radiusB(), raw_radius);
    pending_values_.insert(keys::deltaZ(), gtsam_config_.default_dz);
    pending_graph_.addPrior(
      keys::radiusB(), raw_radius,
      isotropic(1, gtsam_config_.radius_prior_sigma));
    pending_graph_.addPrior(
      keys::deltaZ(), gtsam_config_.default_dz,
      isotropic(1, gtsam_config_.dz_prior_sigma));
  }
}

void Target::addArmorFactors(
  std::uint64_t k,
  const Armor& armor,
  int armor_id,
  const Eigen::Isometry3d& T_world_camera)
{
  const gtsam::Key pose_key = keys::armorPose(k, armor_id);
  const gtsam::Pose3 pose = cameraPose(armor);
  pending_values_.insert(pose_key, pose);

  const gtsam::SharedNoiseModel pixel_noise =
    isotropic(2, gtsam_config_.obs_pixel_sigma);
  for (int point_index = 0; point_index < 4; ++point_index) {
    const cv::Point2f& pixel = armor.points[static_cast<std::size_t>(point_index)];
    pending_graph_.emplace_shared<ArmorReprojFactor>(
      pixel_noise,
      pose_key,
      calibration_.camera_matrix,
      calibration_.distortion_coefficients,
      armor_type_,
      armor_config_,
      point_index,
      Eigen::Vector2d{pixel.x, pixel.y});
  }

  gtsam::Vector4 observation_sigmas;
  observation_sigmas <<
    gtsam_config_.obs_tangential_sigma,
    gtsam_config_.obs_radial_sigma,
    gtsam_config_.obs_height_sigma,
    gtsam_config_.obs_yaw_sigma;
  const gtsam::SharedNoiseModel observation_noise =
    gtsam::noiseModel::Diagonal::Sigmas(observation_sigmas);
  if (armor_count_ != 4 || armor_id % 2 == 0) {
    pending_graph_.emplace_shared<ArmorRadiusCenterZFactor>(
      observation_noise,
      pose_key,
      keys::radiusA(),
      keys::yaw(k),
      keys::center(k),
      T_world_camera,
      armor_id,
      gtsam_config_.radius_min,
      gtsam_config_.radius_max,
      armor_count_);
  } else {
    pending_graph_.emplace_shared<ArmorRadiusDZFactor>(
      observation_noise,
      pose_key,
      keys::radiusB(),
      keys::deltaZ(),
      keys::yaw(k),
      keys::center(k),
      T_world_camera,
      armor_id,
      gtsam_config_.radius_min,
      gtsam_config_.radius_max,
      armor_count_);
  }
}

void Target::commitPendingGraph()
{
  if (failed_) {
    return;
  }
  try {
    isam2_.update(pending_graph_, pending_values_);
    isam_started_ = true;
    pending_graph_.resize(0);
    pending_values_.clear();
    readEstimate();
  } catch (...) {
    failed_ = true;
    pending_graph_.resize(0);
    pending_values_.clear();
    throw;
  }
}

void Target::readEstimate()
{
  const gtsam::Point3 center =
    isam2_.calculateEstimate<gtsam::Point3>(keys::center(k_));
  const gtsam::Vector3 velocity =
    isam2_.calculateEstimate<gtsam::Vector3>(keys::velocity(k_));
  const gtsam::Rot2 yaw =
    isam2_.calculateEstimate<gtsam::Rot2>(keys::yaw(k_));
  const double vyaw = isam2_.calculateEstimate<double>(keys::vyaw(k_));
  const double raw_radius_a =
    isam2_.calculateEstimate<double>(keys::radiusA());
  const double radius_a = logistic(
    raw_radius_a, gtsam_config_.radius_min, gtsam_config_.radius_max);

  state_[CenterX] = center.x();
  state_[VelocityX] = velocity.x();
  state_[CenterY] = center.y();
  state_[VelocityY] = velocity.y();
  state_[CenterZ] = center.z();
  state_[VelocityZ] = velocity.z();
  state_[Yaw] = yaw.theta();
  state_[Vyaw] = vyaw;
  state_[RadiusA] = radius_a;
  state_[RadiusDifference] = 0.0;
  state_[HeightDifference] = 0.0;

  gtsam::KeyVector query{
    keys::center(k_), keys::velocity(k_), keys::yaw(k_),
    keys::vyaw(k_), keys::radiusA()};
  std::vector<int> dimensions{3, 3, 1, 1, 1};
  double raw_radius_b = raw_radius_a;
  if (armor_count_ == 4) {
    raw_radius_b = isam2_.calculateEstimate<double>(keys::radiusB());
    const double radius_b = logistic(
      raw_radius_b, gtsam_config_.radius_min, gtsam_config_.radius_max);
    state_[RadiusDifference] = radius_b - radius_a;
    state_[HeightDifference] =
      isam2_.calculateEstimate<double>(keys::deltaZ());
    query.push_back(keys::radiusB());
    query.push_back(keys::deltaZ());
    dimensions.push_back(1);
    dimensions.push_back(1);
  }

  int raw_dimension = 0;
  for (int dimension : dimensions) {
    raw_dimension += dimension;
  }
  // 分块查询 marginalCovariance，公共快照保留每个图变量的完整块内协方差。
  // 公共快照只要求一份保守可用的 P，因此这里保留每个图变量的完整块内协方差，
  // 跨变量相关项置零，避免把 NaN 泄漏到 L4。
  Eigen::MatrixXd raw_covariance = Eigen::MatrixXd::Zero(raw_dimension, raw_dimension);
  int raw_offset = 0;
  for (std::size_t index = 0; index < query.size(); ++index) {
    raw_covariance.block(
      raw_offset, raw_offset, dimensions[index], dimensions[index]) =
      isam2_.marginalCovariance(query[index]);
    raw_offset += dimensions[index];
  }

  Eigen::MatrixXd transform = Eigen::MatrixXd::Zero(kTargetStateSize, raw_dimension);
  transform(CenterX, 0) = 1.0;
  transform(CenterY, 1) = 1.0;
  transform(CenterZ, 2) = 1.0;
  transform(VelocityX, 3) = 1.0;
  transform(VelocityY, 4) = 1.0;
  transform(VelocityZ, 5) = 1.0;
  transform(Yaw, 6) = 1.0;
  transform(Vyaw, 7) = 1.0;
  const double radius_a_derivative = logisticDerivative(
    radius_a, gtsam_config_.radius_min, gtsam_config_.radius_max);
  transform(RadiusA, 8) = radius_a_derivative;
  if (armor_count_ == 4) {
    const double radius_b = radius_a + state_[RadiusDifference];
    const double radius_b_derivative = logisticDerivative(
      radius_b, gtsam_config_.radius_min, gtsam_config_.radius_max);
    transform(RadiusDifference, 8) = -radius_a_derivative;
    transform(RadiusDifference, 9) = radius_b_derivative;
    transform(HeightDifference, 10) = 1.0;
  }
  covariance_ = transform * raw_covariance * transform.transpose();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose());
  if (!covariance_.allFinite()) {
    throw std::runtime_error("GTSAM joint marginal covariance is not finite");
  }
}

TrackedTarget Target::snapshot() const
{
  TrackedTarget result(
    name_, armor_type_, armor_count_, timestamp_, state_, covariance_, target_config_);
  // JLU 每帧都从当前状态的全部装甲板中选最正对的一块，并没有
  // EKF 的“没见过换板就只信 0 号板”门控。对 GTSAM 快照始终放开选板。
  result.jumped = true;
  result.last_id = last_id_;
  return result;
}

std::vector<Eigen::Vector4d> Target::armorPoses() const
{
  return L3Estimation::armorPoses(state_, armor_count_);
}

bool Target::diverged() const noexcept
{
  if (failed_ || armor_count_ <= 0 || !state_.allFinite() || !covariance_.allFinite()) {
    return true;
  }
  const double radius_a = state_[RadiusA];
  const double radius_b = radius_a + state_[RadiusDifference];
  return radius_a <= gtsam_config_.radius_min || radius_a >= gtsam_config_.radius_max ||
         radius_b <= gtsam_config_.radius_min || radius_b >= gtsam_config_.radius_max;
}

std::size_t Target::activeVariableCount() const noexcept
{
  return isam_started_ ? isam2_.getLinearizationPoint().size()
                       : pending_values_.size();
}

}  // namespace L3Estimation::GtsamEst

#endif  // NEWVISION_USE_GTSAM
