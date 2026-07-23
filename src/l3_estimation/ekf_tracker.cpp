#include "l3_estimation/ekf_tracker.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace L3Estimation {
namespace {

constexpr double kNumericalEpsilon = 1e-9;
constexpr double kCovarianceTolerance = 1e-9;

bool isValidConfig(const EkfTrackerConfig& config) noexcept
{
  return std::isfinite(config.initial_radius)
         && config.initial_variance.allFinite()
         && (config.initial_variance.array() > 0.0).all()
         && std::isfinite(config.linear_acceleration_variance)
         && config.linear_acceleration_variance >= 0.0
         && std::isfinite(config.angular_acceleration_variance)
         && config.angular_acceleration_variance >= 0.0
         && std::isfinite(config.geometry_random_walk_variance)
         && config.geometry_random_walk_variance >= 0.0
         && config.confirmation_hits > 0
         && config.max_predict_interval.count() > 0
         && config.expiration_timeout.count() > 0
         && std::isfinite(config.association_position_gate)
         && config.association_position_gate > 0.0
         && std::isfinite(config.association_yaw_gate)
         && config.association_yaw_gate > 0.0
         && config.association_yaw_gate <= std::numbers::pi
         && std::isfinite(config.nis_gate)
         && config.nis_gate > 0.0
         && std::isfinite(config.min_radius)
         && std::isfinite(config.max_radius)
         && config.min_radius > 0.0
         && config.max_radius > config.min_radius
         && config.initial_radius >= config.min_radius
         && config.initial_radius <= config.max_radius
         && std::isfinite(config.max_abs_height_offset)
         && config.max_abs_height_offset >= 0.0;
}

bool isValidObservation(
  const ArmorObservation& observation,
  int robot_id) noexcept
{
  return observation.robot_id == robot_id
         && observation.position_world.allFinite()
         && std::isfinite(observation.yaw_world)
         && std::isfinite(observation.confidence)
         && observation.confidence >= 0.0F
         && std::isfinite(observation.reprojection_error_px)
         && observation.reprojection_error_px >= 0.0;
}

}  // namespace

EkfTracker::EkfTracker(int robot_id, EkfTrackerConfig config)
  : robot_id_(robot_id), config_(std::move(config))
{
  if (robot_id_ < -1) {
    throw std::invalid_argument("EkfTracker received an invalid robot_id");
  }
  if (!isValidConfig(config_)) {
    throw std::invalid_argument("EkfTracker received an invalid configuration");
  }
  reset();
}

double EkfTracker::normalizeAngle(double angle) noexcept
{
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  angle = std::remainder(angle, kTwoPi);
  return angle <= -std::numbers::pi ? angle + kTwoPi : angle;
}

void EkfTracker::reset()
{
  x_.setZero();
  covariance_.setIdentity();
  state_ = TargetState{};
  state_.robot_id = robot_id_;
  tracker_state_ = TrackerState::Lost;
  filter_time_ = TimePoint{};
  last_seen_ = TimePoint{};
  successful_frame_count_ = 0;
  initialized_ = false;
  awaiting_update_ = false;
}

void EkfTracker::initialize(const ArmorObservation& observation)
{
  const double yaw = normalizeAngle(observation.yaw_world);

  x_.setZero();
  x_[XC] = observation.position_world.x()
           + config_.initial_radius * std::cos(yaw);
  x_[YC] = observation.position_world.y()
           + config_.initial_radius * std::sin(yaw);
  x_[ZC] = observation.position_world.z();
  x_[YAW] = yaw;
  x_[RADIUS] = config_.initial_radius;

  covariance_ = config_.initial_variance.asDiagonal();
  filter_time_ = observation.timestamp;
  last_seen_ = observation.timestamp;
  successful_frame_count_ = 1;
  initialized_ = true;
  awaiting_update_ = false;
  tracker_state_ = config_.confirmation_hits <= 1
                     ? TrackerState::Tracking
                     : TrackerState::Detecting;
  publishState();
}

StateCovariance EkfTracker::buildTransition(double dt) const noexcept
{
  StateCovariance transition = StateCovariance::Identity();
  transition(XC, VX) = dt;
  transition(YC, VY) = dt;
  transition(ZC, VZ) = dt;
  transition(YAW, YAW_RATE) = dt;
  return transition;
}

StateCovariance EkfTracker::buildProcessNoise(double dt) const noexcept
{
  StateCovariance process_noise = StateCovariance::Zero();

  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;

  const auto fill_white_noise_block = [&process_noise, dt2, dt3, dt4](
      int position_index,
      int velocity_index,
      double acceleration_variance) {
    process_noise(position_index, position_index) =
      0.25 * dt4 * acceleration_variance;
    process_noise(position_index, velocity_index) =
      0.5 * dt3 * acceleration_variance;
    process_noise(velocity_index, position_index) =
      0.5 * dt3 * acceleration_variance;
    process_noise(velocity_index, velocity_index) =
      dt2 * acceleration_variance;
  };

  fill_white_noise_block(
    XC, VX, config_.linear_acceleration_variance);
  fill_white_noise_block(
    YC, VY, config_.linear_acceleration_variance);
  fill_white_noise_block(
    ZC, VZ, config_.linear_acceleration_variance);
  fill_white_noise_block(
    YAW, YAW_RATE, config_.angular_acceleration_variance);

  const double geometry_noise =
    config_.geometry_random_walk_variance * dt;
  process_noise(RADIUS, RADIUS) = geometry_noise;
  process_noise(RADIUS_OFFSET, RADIUS_OFFSET) = geometry_noise;
  process_noise(HEIGHT_OFFSET, HEIGHT_OFFSET) = geometry_noise;
  return process_noise;
}

void EkfTracker::predict(TimePoint timestamp)
{
  if (!initialized_) {
    return;
  }

  if (timestamp < filter_time_) {
    reset();
    return;
  }
  if (timestamp == filter_time_) {
    return;
  }

  if (awaiting_update_ && tracker_state_ == TrackerState::Detecting) {
    successful_frame_count_ = 0;
  }
  awaiting_update_ = true;

  const double dt = std::chrono::duration<double>(timestamp - filter_time_).count();
  const double max_dt =
    std::chrono::duration<double>(config_.max_predict_interval).count();
  if (!std::isfinite(dt) || dt <= 0.0 || dt > max_dt) {
    reset();
    return;
  }

  const StateCovariance transition = buildTransition(dt);
  StateVector predicted_state = transition * x_;
  predicted_state[YAW] = normalizeAngle(predicted_state[YAW]);

  StateCovariance predicted_covariance =
    transition * covariance_ * transition.transpose()
    + buildProcessNoise(dt);
  predicted_covariance =
    0.5 * (predicted_covariance + predicted_covariance.transpose());

  if (!validateState(predicted_state, predicted_covariance)) {
    reset();
    return;
  }

  x_ = predicted_state;
  covariance_ = predicted_covariance;
  filter_time_ = timestamp;
  if (tracker_state_ == TrackerState::Tracking) {
    tracker_state_ = TrackerState::TemporaryLost;
  }
  publishState();
}

Eigen::Vector3d EkfTracker::armorPosition(
  const StateVector& state,
  int face_id) const noexcept
{
  const double angle = normalizeAngle(
    state[YAW] + static_cast<double>(face_id) * std::numbers::pi / 2.0);
  const bool second_group = face_id % 2 != 0;
  const double radius = state[RADIUS]
                        + (second_group ? state[RADIUS_OFFSET] : 0.0);

  return {
    state[XC] - radius * std::cos(angle),
    state[YC] - radius * std::sin(angle),
    state[ZC] + (second_group ? state[HEIGHT_OFFSET] : 0.0)};
}

EkfTracker::MeasurementVector EkfTracker::predictMeasurement(
  const StateVector& state,
  int face_id) const
{
  const Eigen::Vector3d position = armorPosition(state, face_id);
  const double horizontal_distance = std::hypot(position.x(), position.y());
  const double distance = position.norm();
  const double armor_yaw = normalizeAngle(
    state[YAW] + static_cast<double>(face_id) * std::numbers::pi / 2.0);

  return {
    std::atan2(position.y(), position.x()),
    std::atan2(position.z(), horizontal_distance),
    distance,
    armor_yaw};
}

EkfTracker::MeasurementJacobian EkfTracker::measurementJacobian(
  const StateVector& state,
  int face_id) const
{
  const double angle = normalizeAngle(
    state[YAW] + static_cast<double>(face_id) * std::numbers::pi / 2.0);
  const bool second_group = face_id % 2 != 0;
  const double radius = state[RADIUS]
                        + (second_group ? state[RADIUS_OFFSET] : 0.0);

  Eigen::Matrix<double, 3, STATE_DIM> position_jacobian =
    Eigen::Matrix<double, 3, STATE_DIM>::Zero();
  position_jacobian(0, XC) = 1.0;
  position_jacobian(0, YAW) = radius * std::sin(angle);
  position_jacobian(0, RADIUS) = -std::cos(angle);
  position_jacobian(1, YC) = 1.0;
  position_jacobian(1, YAW) = -radius * std::cos(angle);
  position_jacobian(1, RADIUS) = -std::sin(angle);
  position_jacobian(2, ZC) = 1.0;

  if (second_group) {
    position_jacobian(0, RADIUS_OFFSET) = -std::cos(angle);
    position_jacobian(1, RADIUS_OFFSET) = -std::sin(angle);
    position_jacobian(2, HEIGHT_OFFSET) = 1.0;
  }

  const Eigen::Vector3d position = armorPosition(state, face_id);
  const double x = position.x();
  const double y = position.y();
  const double z = position.z();
  const double horizontal_squared = x * x + y * y;
  const double horizontal = std::sqrt(horizontal_squared);
  const double distance_squared = horizontal_squared + z * z;
  const double distance = std::sqrt(distance_squared);

  if (horizontal_squared <= kNumericalEpsilon
      || distance_squared <= kNumericalEpsilon) {
    return MeasurementJacobian::Constant(
      std::numeric_limits<double>::quiet_NaN());
  }

  Eigen::Matrix3d spherical_jacobian = Eigen::Matrix3d::Zero();
  spherical_jacobian(0, 0) = -y / horizontal_squared;
  spherical_jacobian(0, 1) = x / horizontal_squared;
  spherical_jacobian(1, 0) = -x * z / (horizontal * distance_squared);
  spherical_jacobian(1, 1) = -y * z / (horizontal * distance_squared);
  spherical_jacobian(1, 2) = horizontal / distance_squared;
  spherical_jacobian(2, 0) = x / distance;
  spherical_jacobian(2, 1) = y / distance;
  spherical_jacobian(2, 2) = z / distance;

  MeasurementJacobian measurement_jacobian = MeasurementJacobian::Zero();
  measurement_jacobian.template topRows<3>() =
    spherical_jacobian * position_jacobian;
  measurement_jacobian(3, YAW) = 1.0;
  return measurement_jacobian;
}

EkfTracker::MeasurementVector EkfTracker::observationMeasurement(
  const ArmorObservation& observation) noexcept
{
  const Eigen::Vector3d& position = observation.position_world;
  return {
    std::atan2(position.y(), position.x()),
    std::atan2(position.z(), std::hypot(position.x(), position.y())),
    position.norm(),
    normalizeAngle(observation.yaw_world)};
}

EkfTracker::MeasurementVector EkfTracker::measurementResidual(
  const MeasurementVector& measured,
  const MeasurementVector& predicted) noexcept
{
  MeasurementVector residual = measured - predicted;
  residual[0] = normalizeAngle(residual[0]);
  residual[1] = normalizeAngle(residual[1]);
  residual[3] = normalizeAngle(residual[3]);
  return residual;
}

EkfTracker::MeasurementCovariance EkfTracker::measurementNoise(
  const ArmorObservation& observation) const noexcept
{
  constexpr double kBearingStandardDeviation = 0.01;
  constexpr double kArmorYawStandardDeviation = 0.15;

  const double distance = observation.position_world.norm();
  const double distance_standard_deviation = 0.05 + 0.01 * distance * distance;
  double quality_scale =
    1.0 / std::clamp(static_cast<double>(observation.confidence), 0.25, 1.0);
  if (observation.reprojection_error_px > 0.0) {
    quality_scale *= 1.0
                     + observation.reprojection_error_px
                         * observation.reprojection_error_px;
  }

  MeasurementCovariance noise = MeasurementCovariance::Zero();
  noise(0, 0) = kBearingStandardDeviation * kBearingStandardDeviation
                * quality_scale;
  noise(1, 1) = kBearingStandardDeviation * kBearingStandardDeviation
                * quality_scale;
  noise(2, 2) = distance_standard_deviation * distance_standard_deviation
                * quality_scale;
  noise(3, 3) = kArmorYawStandardDeviation * kArmorYawStandardDeviation
                * quality_scale;
  return noise;
}

std::vector<EkfTracker::Association> EkfTracker::associateObservations(
  const std::vector<ArmorObservation>& observations,
  const std::vector<std::size_t>& observation_indices,
  unsigned int reserved_face_mask) const
{
  struct Candidate {
    bool valid = false;
    double nis = std::numeric_limits<double>::infinity();
  };

  std::vector<std::array<Candidate, kArmorFaceCount>> candidates(
    observation_indices.size());

  for (std::size_t local_index = 0;
       local_index < observation_indices.size();
       ++local_index) {
    const auto& observation = observations[observation_indices[local_index]];
    const MeasurementVector measured = observationMeasurement(observation);
    const MeasurementCovariance noise = measurementNoise(observation);

    for (int face_id = 0; face_id < kArmorFaceCount; ++face_id) {
      if ((reserved_face_mask & (1U << face_id)) != 0U) {
        continue;
      }

      const Eigen::Vector3d predicted_position = armorPosition(x_, face_id);
      const double position_error =
        (observation.position_world - predicted_position).norm();
      const double predicted_yaw = normalizeAngle(
        x_[YAW] + static_cast<double>(face_id) * std::numbers::pi / 2.0);
      const double yaw_error = std::abs(normalizeAngle(
        observation.yaw_world - predicted_yaw));
      if (position_error > config_.association_position_gate
          || yaw_error > config_.association_yaw_gate) {
        continue;
      }

      const MeasurementJacobian jacobian = measurementJacobian(x_, face_id);
      if (!jacobian.allFinite()) {
        continue;
      }

      const MeasurementVector residual = measurementResidual(
        measured, predictMeasurement(x_, face_id));
      const MeasurementCovariance innovation_covariance =
        jacobian * covariance_ * jacobian.transpose() + noise;
      Eigen::LDLT<MeasurementCovariance> decomposition(
        innovation_covariance);
      if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
        continue;
      }

      const MeasurementVector solved_residual = decomposition.solve(residual);
      const double nis = residual.dot(solved_residual);
      if (!std::isfinite(nis) || nis < 0.0 || nis > config_.nis_gate) {
        continue;
      }
      candidates[local_index][face_id] = Candidate{true, nis};
    }
  }

  constexpr std::size_t kMaskCount = 1U << kArmorFaceCount;
  struct AssignmentState {
    double cost = std::numeric_limits<double>::infinity();
    std::vector<Association> associations;
  };

  std::array<AssignmentState, kMaskCount> states;
  states[reserved_face_mask].cost = 0.0;

  for (std::size_t local_index = 0;
       local_index < observation_indices.size();
       ++local_index) {
    auto next_states = states;
    for (std::size_t mask = 0; mask < kMaskCount; ++mask) {
      if (!std::isfinite(states[mask].cost)) {
        continue;
      }
      for (int face_id = 0; face_id < kArmorFaceCount; ++face_id) {
        if ((mask & (1U << face_id)) != 0U
            || !candidates[local_index][face_id].valid) {
          continue;
        }

        const std::size_t next_mask = mask | (1U << face_id);
        const double next_cost =
          states[mask].cost + candidates[local_index][face_id].nis;
        if (next_cost >= next_states[next_mask].cost) {
          continue;
        }

        next_states[next_mask] = states[mask];
        next_states[next_mask].cost = next_cost;
        next_states[next_mask].associations.push_back(Association{
          observation_indices[local_index],
          face_id,
          candidates[local_index][face_id].nis});
      }
    }
    states = std::move(next_states);
  }

  const AssignmentState* best = nullptr;
  int best_count = -1;
  for (std::size_t mask = 0; mask < kMaskCount; ++mask) {
    if (!std::isfinite(states[mask].cost)) {
      continue;
    }
    const int assigned_count =
      std::popcount(static_cast<unsigned int>(mask & ~reserved_face_mask));
    if (assigned_count > best_count
        || (assigned_count == best_count
            && best != nullptr
            && states[mask].cost < best->cost)) {
      best = &states[mask];
      best_count = assigned_count;
    }
  }

  if (best == nullptr) {
    return {};
  }

  std::vector<Association> result = best->associations;
  std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.face_id < rhs.face_id;
  });
  return result;
}

bool EkfTracker::correct(
  const ArmorObservation& observation,
  int face_id)
{
  const MeasurementJacobian jacobian = measurementJacobian(x_, face_id);
  if (!jacobian.allFinite()) {
    return false;
  }

  const MeasurementCovariance noise = measurementNoise(observation);
  const MeasurementVector residual = measurementResidual(
    observationMeasurement(observation),
    predictMeasurement(x_, face_id));
  const MeasurementCovariance innovation_covariance =
    jacobian * covariance_ * jacobian.transpose() + noise;

  Eigen::LDLT<MeasurementCovariance> decomposition(innovation_covariance);
  if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
    return false;
  }

  const MeasurementVector solved_residual = decomposition.solve(residual);
  const double nis = residual.dot(solved_residual);
  if (!std::isfinite(nis) || nis < 0.0 || nis > config_.nis_gate) {
    return false;
  }

  const Eigen::Matrix<double, STATE_DIM, kMeasurementDim> kalman_gain =
    decomposition.solve(jacobian * covariance_).transpose();
  StateVector corrected_state = x_ + kalman_gain * residual;
  corrected_state[YAW] = normalizeAngle(corrected_state[YAW]);

  const StateCovariance identity = StateCovariance::Identity();
  const StateCovariance correction = identity - kalman_gain * jacobian;
  StateCovariance corrected_covariance =
    correction * covariance_ * correction.transpose()
    + kalman_gain * noise * kalman_gain.transpose();
  corrected_covariance =
    0.5 * (corrected_covariance + corrected_covariance.transpose());

  if (!validateState(corrected_state, corrected_covariance)) {
    return false;
  }

  x_ = corrected_state;
  covariance_ = corrected_covariance;
  return true;
}

bool EkfTracker::validateState(
  const StateVector& state,
  const StateCovariance& covariance) const
{
  if (!state.allFinite() || !covariance.allFinite()) {
    return false;
  }

  const double first_radius = state[RADIUS];
  const double second_radius = state[RADIUS] + state[RADIUS_OFFSET];
  if (first_radius < config_.min_radius
      || first_radius > config_.max_radius
      || second_radius < config_.min_radius
      || second_radius > config_.max_radius
      || std::abs(state[HEIGHT_OFFSET]) > config_.max_abs_height_offset) {
    return false;
  }

  const StateCovariance symmetric_covariance =
    0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<StateCovariance> eigen_solver(
    symmetric_covariance,
    Eigen::EigenvaluesOnly);
  return eigen_solver.info() == Eigen::Success
         && eigen_solver.eigenvalues().minCoeff() >= -kCovarianceTolerance;
}

void EkfTracker::publishState()
{
  state_ = TargetState{};
  state_.robot_id = robot_id_;
  if (!initialized_
      || (tracker_state_ != TrackerState::Tracking
          && tracker_state_ != TrackerState::TemporaryLost)) {
    return;
  }

  state_.center = {x_[XC], x_[YC], x_[ZC]};
  state_.velocity = {x_[VX], x_[VY], x_[VZ]};
  state_.yaw = normalizeAngle(x_[YAW]);
  state_.yaw_rate = x_[YAW_RATE];
  state_.radius = x_[RADIUS];
  state_.radius_offset = x_[RADIUS_OFFSET];
  state_.height_offset = x_[HEIGHT_OFFSET];
  state_.covariance = covariance_;
  state_.timestamp = filter_time_;
}

void EkfTracker::update(
  const std::vector<ArmorObservation>& observations)
{
  if (observations.empty()) {
    return;
  }

  TimePoint update_time{};
  bool has_valid_observation = false;
  for (const auto& observation : observations) {
    if (!isValidObservation(observation, robot_id_)) {
      continue;
    }
    if (!has_valid_observation || observation.timestamp > update_time) {
      update_time = observation.timestamp;
      has_valid_observation = true;
    }
  }
  if (!has_valid_observation) {
    return;
  }

  if (initialized_ && update_time < filter_time_) {
    return;
  }
  if (initialized_ && update_time > filter_time_) {
    predict(update_time);
  }

  std::vector<std::size_t> valid_indices;
  valid_indices.reserve(observations.size());
  for (std::size_t index = 0; index < observations.size(); ++index) {
    if (observations[index].timestamp == update_time
        && isValidObservation(observations[index], robot_id_)) {
      valid_indices.push_back(index);
    }
  }
  if (valid_indices.empty()) {
    return;
  }

  const bool is_new_observation_frame =
    !initialized_ || update_time > last_seen_;

  bool initialized_this_frame = false;
  std::size_t anchor_index = std::numeric_limits<std::size_t>::max();
  if (!initialized_) {
    anchor_index = *std::max_element(
      valid_indices.begin(), valid_indices.end(),
      [&observations](std::size_t lhs, std::size_t rhs) {
        const auto& left = observations[lhs];
        const auto& right = observations[rhs];
        if (left.confidence != right.confidence) {
          return left.confidence < right.confidence;
        }
        return left.reprojection_error_px > right.reprojection_error_px;
      });
    initialize(observations[anchor_index]);
    initialized_this_frame = true;
  }

  std::vector<std::size_t> association_indices;
  association_indices.reserve(valid_indices.size());
  for (const std::size_t index : valid_indices) {
    if (index != anchor_index) {
      association_indices.push_back(index);
    }
  }

  const unsigned int reserved_face_mask =
    initialized_this_frame ? 1U : 0U;
  const auto associations = associateObservations(
    observations, association_indices, reserved_face_mask);

  bool accepted_observation = initialized_this_frame;
  for (const auto& association : associations) {
    accepted_observation =
      correct(observations[association.observation_index], association.face_id)
      || accepted_observation;
  }

  if (!accepted_observation) {
    publishState();
    return;
  }

  last_seen_ = update_time;
  awaiting_update_ = false;
  if (!initialized_this_frame && is_new_observation_frame) {
    if (tracker_state_ == TrackerState::Detecting) {
      ++successful_frame_count_;
      if (successful_frame_count_ >= config_.confirmation_hits) {
        tracker_state_ = TrackerState::Tracking;
      }
    } else {
      tracker_state_ = TrackerState::Tracking;
    }
  }
  publishState();
}

const TargetState& EkfTracker::state() const noexcept
{
  return state_;
}

TrackerState EkfTracker::trackerState() const noexcept
{
  return tracker_state_;
}

bool EkfTracker::expired(TimePoint now) const noexcept
{
  if (!initialized_ || tracker_state_ == TrackerState::Lost
      || now < last_seen_) {
    return true;
  }
  return now - last_seen_ > config_.expiration_timeout;
}

}  // namespace L3Estimation
