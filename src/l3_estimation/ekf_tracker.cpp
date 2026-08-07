#include "l3_estimation/ekf_tracker.hpp"
#include "l3_estimation/angle_utils.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace L3Estimation {
namespace {

constexpr double kCovarianceTolerance = 1e-9;
constexpr double kGeometryTolerance = 1e-12;

// 配置合法性检查（范围、有限性、一致性）。
bool validConfig(const EkfTrackerConfig& config) noexcept
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
         && std::isfinite(config.position_standard_deviation_base_m)
         && config.position_standard_deviation_base_m > 0.0
         && std::isfinite(config.position_standard_deviation_quadratic)
         && config.position_standard_deviation_quadratic >= 0.0
         && std::isfinite(config.armor_yaw_standard_deviation_rad)
         && config.armor_yaw_standard_deviation_rad > 0.0
         && config.confirmation_hits > 0
         && config.max_predict_interval.count() > 0
         && config.expiration_timeout.count() > 0
         && std::isfinite(config.association_position_gate)
         && config.association_position_gate > 0.0
         && std::isfinite(config.association_yaw_gate)
         && config.association_yaw_gate > 0.0
         && config.association_yaw_gate <= std::numbers::pi
         && std::isfinite(config.association_radius_gate)
         && config.association_radius_gate > 0.0
         && std::isfinite(config.association_position_weight)
         && config.association_position_weight >= 0.0
         && std::isfinite(config.association_yaw_weight)
         && config.association_yaw_weight >= 0.0
         && std::isfinite(config.association_radius_weight)
         && config.association_radius_weight >= 0.0
         && std::isfinite(config.nis_reference_threshold)
         && config.nis_reference_threshold > 0.0
         && std::isfinite(config.min_radius)
         && std::isfinite(config.max_radius)
         && config.min_radius > 0.0
         && config.max_radius > config.min_radius
         && config.initial_radius >= config.min_radius
         && config.initial_radius <= config.max_radius
         && std::isfinite(config.minimum_four_armor_corner_angle_rad)
         && config.minimum_four_armor_corner_angle_rad > 0.0
         && config.minimum_four_armor_corner_angle_rad
              <= std::numbers::pi / 2.0
         && std::isfinite(config.max_abs_height_offset)
         && config.max_abs_height_offset >= 0.0;
}

// 观测字段基本校验：robot/model 匹配、有限且非负。
bool validObservation(
  const ArmorObservation& observation,
  int robot_id,
  TargetModel model) noexcept
{
  return observation.robot_id == robot_id
         && observation.model == model
         && observation.position_world.allFinite()
         && std::isfinite(observation.yaw_world)
         && std::isfinite(observation.confidence)
         && observation.confidence >= 0.0F
         && std::isfinite(observation.optimized_reprojection_error_px)
         && observation.optimized_reprojection_error_px >= 0.0;
}

}  // namespace

EkfTracker::EkfTracker(int robot_id, EkfTrackerConfig config)
  : EkfTracker(
      robot_id,
      TargetModel::FourArmorVehicle,
      std::move(config))
{
}

EkfTracker::EkfTracker(
  int robot_id,
  TargetModel model,
  EkfTrackerConfig config)
  : robot_id_(robot_id),
    model_(model),
    config_(std::move(config))
{
  if (robot_id_ < -1
      || (model_ != TargetModel::FourArmorVehicle
          && model_ != TargetModel::ThreeArmorOutpost)
      || !validConfig(config_)) {
    throw std::invalid_argument(
      "EkfTracker received an invalid id, model or configuration");
  }
  reset();
}

void EkfTracker::reset()
{
  // 重置到 Lost，清空状态并发布空状态。
  x_.setZero();
  covariance_.setIdentity();
  tracker_state_ = TrackerState::Lost;
  filter_time_ = TimePoint{};
  last_seen_ = TimePoint{};
  successful_frame_count_ = 0;
  initialized_ = false;
  updated_this_frame_ = false;
  last_quality_ = {};
  publishState();
}

int EkfTracker::armorFaceCount() const noexcept
{
  return targetModelTraits(model_).armor_count;
}

double EkfTracker::faceAngle(int face_id) const noexcept
{
  return static_cast<double>(face_id)
         * targetModelTraits(model_).face_angle_interval_rad;
}

bool EkfTracker::usesSecondGeometryGroup(int face_id) const noexcept
{
  return targetModelTraits(model_).uses_alternating_radius_and_height
         && face_id % 2 != 0;
}

void EkfTracker::initialize(const ArmorObservation& observation)
{
  // 用首块观测定义 0 号面，由初始半径反推旋转中心。
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
  applyModelConstraints(x_, covariance_);

  filter_time_ = observation.timestamp;
  last_seen_ = observation.timestamp;
  successful_frame_count_ = 1;
  initialized_ = true;
  updated_this_frame_ = true;
  tracker_state_ = config_.confirmation_hits <= 1
                     ? TrackerState::Tracking
                     : TrackerState::Detecting;
}

StateCovariance EkfTracker::buildTransition(double dt) const noexcept
{
  // 匀速 + 匀角速度的状态转移矩阵。
  StateCovariance transition = StateCovariance::Identity();
  transition(XC, VX) = dt;
  transition(YC, VY) = dt;
  transition(ZC, VZ) = dt;
  transition(YAW, YAW_RATE) = dt;
  return transition;
}

StateCovariance EkfTracker::buildProcessNoise(double dt) const noexcept
{
  // 位置/yaw 分段白噪声加速度 + 几何量随机游走的过程噪声。
  StateCovariance process_noise = StateCovariance::Zero();
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;

  // 位置和 yaw 都使用分段白噪声加速度模型。
  const auto fill_block = [&process_noise, dt2, dt3, dt4](
      int position,
      int velocity,
      double variance) {
    process_noise(position, position) = 0.25 * dt4 * variance;
    process_noise(position, velocity) = 0.5 * dt3 * variance;
    process_noise(velocity, position) = 0.5 * dt3 * variance;
    process_noise(velocity, velocity) = dt2 * variance;
  };
  fill_block(XC, VX, config_.linear_acceleration_variance);
  fill_block(YC, VY, config_.linear_acceleration_variance);
  fill_block(ZC, VZ, config_.linear_acceleration_variance);
  fill_block(YAW, YAW_RATE, config_.angular_acceleration_variance);

  const double geometry_noise =
    config_.geometry_random_walk_variance * dt;
  process_noise(RADIUS, RADIUS) = geometry_noise;
  if (model_ == TargetModel::FourArmorVehicle) {
    process_noise(RADIUS_OFFSET, RADIUS_OFFSET) = geometry_noise;
    process_noise(HEIGHT_OFFSET, HEIGHT_OFFSET) = geometry_noise;
  }
  return process_noise;
}

void EkfTracker::predict(TimePoint timestamp)
{
  // 推进到 timestamp；时间倒退、超间隔或状态非法时安全重置。
  updated_this_frame_ = false;
  if (!initialized_) {
    publishState();
    return;
  }
  if (timestamp < filter_time_) {
    reset();
    return;
  }
  if (timestamp == filter_time_) {
    publishState();
    return;
  }

  const double dt =
    std::chrono::duration<double>(timestamp - filter_time_).count();
  const double maximum_dt =
    std::chrono::duration<double>(
      config_.max_predict_interval).count();
  if (!std::isfinite(dt) || dt <= 0.0 || dt > maximum_dt) {
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
  applyModelConstraints(predicted_state, predicted_covariance);
  if (!validateState(predicted_state, predicted_covariance)) {
    reset();
    return;
  }

  x_ = predicted_state;
  covariance_ = predicted_covariance;
  filter_time_ = timestamp;
  publishState();
}

Eigen::Vector3d EkfTracker::armorPosition(
  const StateVector& state,
  int face_id) const noexcept
{
  // 第 face_id 面装甲中心的预测位置。
  const double angle = normalizeAngle(state[YAW] + faceAngle(face_id));
  const bool second_group = usesSecondGeometryGroup(face_id);
  const double radius =
    state[RADIUS]
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
  return {
    position.x(),
    position.y(),
    position.z(),
    normalizeAngle(state[YAW] + faceAngle(face_id))};
}

EkfTracker::MeasurementJacobian EkfTracker::measurementJacobian(
  const StateVector& state,
  int face_id) const
{
  const double angle = normalizeAngle(state[YAW] + faceAngle(face_id));
  const bool second_group = usesSecondGeometryGroup(face_id);
  const double radius =
    state[RADIUS]
    + (second_group ? state[RADIUS_OFFSET] : 0.0);

  // h(x)=[armor_x,armor_y,armor_z,face_yaw] 的直接 Jacobian。
  MeasurementJacobian jacobian = MeasurementJacobian::Zero();
  jacobian(0, XC) = 1.0;
  jacobian(0, YAW) = radius * std::sin(angle);
  jacobian(0, RADIUS) = -std::cos(angle);
  jacobian(1, YC) = 1.0;
  jacobian(1, YAW) = -radius * std::cos(angle);
  jacobian(1, RADIUS) = -std::sin(angle);
  jacobian(2, ZC) = 1.0;
  jacobian(3, YAW) = 1.0;
  if (second_group) {
    jacobian(0, RADIUS_OFFSET) = -std::cos(angle);
    jacobian(1, RADIUS_OFFSET) = -std::sin(angle);
    jacobian(2, HEIGHT_OFFSET) = 1.0;
  }
  return jacobian;
}

EkfTracker::MeasurementCovariance EkfTracker::measurementNoise(
  const ArmorObservation& observation) const noexcept
{
  // 距离相关的位置噪声与 yaw 噪声，按检测置信度缩放。
  const double distance = observation.position_world.norm();
  const double position_standard_deviation =
    config_.position_standard_deviation_base_m
    + config_.position_standard_deviation_quadratic
        * distance * distance;
  const double confidence_scale =
    1.0 / std::clamp(
      static_cast<double>(observation.confidence),
      0.25,
      1.0);

  MeasurementCovariance noise = MeasurementCovariance::Zero();
  const double position_variance =
    position_standard_deviation * position_standard_deviation
    * confidence_scale;
  noise(0, 0) = position_variance;
  noise(1, 1) = position_variance;
  noise(2, 2) = position_variance;
  noise(3, 3) =
    config_.armor_yaw_standard_deviation_rad
    * config_.armor_yaw_standard_deviation_rad
    * confidence_scale;
  return noise;
}

EkfTracker::MeasurementVector EkfTracker::observationMeasurement(
  const ArmorObservation& observation) noexcept
{
  return {
    observation.position_world.x(),
    observation.position_world.y(),
    observation.position_world.z(),
    normalizeAngle(observation.yaw_world)};
}

EkfTracker::MeasurementVector EkfTracker::measurementResidual(
  const MeasurementVector& measured,
  const MeasurementVector& predicted) noexcept
{
  MeasurementVector residual = measured - predicted;
  residual[3] = normalizeAngle(residual[3]);
  return residual;
}

std::vector<EkfTracker::Association> EkfTracker::associateObservations(
  const std::vector<ArmorObservation>& observations,
  const std::vector<std::size_t>& observation_indices) const
{
  // 对每块观测枚举所有物理面：先按位置/yaw 门限和几何约束生成候选，
  // 代价用各自门限归一化（位置、yaw、半径加权平方和）；最后按代价排序，
  // 几何约束开启时保证一块观测和一个物理面最多使用一次。
  std::vector<Association> candidates;
  for (const std::size_t observation_index : observation_indices) {
    const auto& observation = observations[observation_index];
    for (int face_id = 0; face_id < armorFaceCount(); ++face_id) {
      const Eigen::Vector3d predicted_position =
        armorPosition(x_, face_id);
      const double position_error =
        (observation.position_world - predicted_position).norm();
      const double predicted_yaw =
        normalizeAngle(x_[YAW] + faceAngle(face_id));
      const double yaw_error = std::abs(normalizeAngle(
        observation.yaw_world - predicted_yaw));
      const double expected_radius = x_[RADIUS]
        + (usesSecondGeometryGroup(face_id) ? x_[RADIUS_OFFSET] : 0.0);
      const Eigen::Vector2d face_normal{
        std::cos(predicted_yaw), std::sin(predicted_yaw)};
      const Eigen::Vector2d center_to_observation{
        x_[XC] - observation.position_world.x(),
        x_[YC] - observation.position_world.y()};
      const double implied_radius =
        center_to_observation.dot(face_normal);
      const double radius_error =
        std::abs(implied_radius - expected_radius);

      double minimum_corner_angle = 0.0;
      if (model_ == TargetModel::FourArmorVehicle) {
        const double candidate_first_radius =
          usesSecondGeometryGroup(face_id) ? x_[RADIUS] : implied_radius;
        const double candidate_second_radius =
          usesSecondGeometryGroup(face_id)
            ? implied_radius
            : x_[RADIUS] + x_[RADIUS_OFFSET];
        minimum_corner_angle = fourArmorMinimumCornerAngle(
          candidate_first_radius, candidate_second_radius);
      }
      if (!std::isfinite(position_error)
          || !std::isfinite(yaw_error)
          || position_error > config_.association_position_gate
          || yaw_error > config_.association_yaw_gate
          || (config_.enable_vehicle_geometry_constraints
              && (!std::isfinite(implied_radius)
                  || !std::isfinite(radius_error)
                  || implied_radius < config_.min_radius
                  || implied_radius > config_.max_radius
                  || radius_error > config_.association_radius_gate
                  || (model_ == TargetModel::FourArmorVehicle
                      && minimum_corner_angle + kGeometryTolerance
                           < config_.minimum_four_armor_corner_angle_rad)))) {
        continue;
      }

      const double normalized_position =
        position_error / config_.association_position_gate;
      const double normalized_yaw =
        yaw_error / config_.association_yaw_gate;
      const double normalized_radius =
        radius_error / config_.association_radius_gate;
      const double cost =
        config_.association_position_weight
          * normalized_position * normalized_position
        + config_.association_yaw_weight
          * normalized_yaw * normalized_yaw
        + (config_.enable_vehicle_geometry_constraints
             ? config_.association_radius_weight
                 * normalized_radius * normalized_radius
             : 0.0);
      candidates.push_back({
        .observation_index = observation_index,
        .face_id = face_id,
        .position_error_m = position_error,
        .yaw_error_rad = yaw_error,
        .implied_radius_m = implied_radius,
        .radius_error_m = radius_error,
        .minimum_corner_angle_rad = minimum_corner_angle,
        .match_cost = cost});
    }
  }

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const auto& lhs, const auto& rhs) {
      return lhs.match_cost < rhs.match_cost;
    });

  // 几何约束开启时，一个物理面只能使用一次；关闭时保留原基线行为，
  // 每块观测独立选择最低代价物理面，便于同数据 A/B 对比。
  std::vector<Association> associations;
  std::vector<bool> observation_used(observations.size(), false);
  std::vector<bool> face_used(
    static_cast<std::size_t>(armorFaceCount()), false);
  for (const auto& candidate : candidates) {
    if (observation_used[candidate.observation_index]
        || (config_.enable_vehicle_geometry_constraints
            && face_used[static_cast<std::size_t>(candidate.face_id)])) {
      continue;
    }
    observation_used[candidate.observation_index] = true;
    if (config_.enable_vehicle_geometry_constraints) {
      face_used[static_cast<std::size_t>(candidate.face_id)] = true;
    }
    associations.push_back(candidate);
  }
  return associations;
}

AssociationDiagnostic EkfTracker::correct(
  const ArmorObservation& observation,
  const Association& association)
{
  // 单块观测对当前状态的 EKF 校正；Joseph 形式，NIS 只记录不拒绝。
  AssociationDiagnostic diagnostic{
    .observation_index = association.observation_index,
    .robot_id = robot_id_,
    .associated_face_id = association.face_id,
    .position_error_m = association.position_error_m,
    .yaw_error_rad = association.yaw_error_rad,
    .implied_radius_m = association.implied_radius_m,
    .radius_error_m = association.radius_error_m,
    .minimum_corner_angle_rad = association.minimum_corner_angle_rad,
    .match_cost = association.match_cost,
    .lifecycle_before = tracker_state_,
    .lifecycle_after = tracker_state_};

  const MeasurementJacobian jacobian =
    measurementJacobian(x_, association.face_id);
  const MeasurementCovariance noise = measurementNoise(observation);
  const MeasurementVector residual = measurementResidual(
    observationMeasurement(observation),
    predictMeasurement(x_, association.face_id));
  if (!jacobian.allFinite()
      || !noise.allFinite()
      || !residual.allFinite()) {
    return diagnostic;
  }

  const MeasurementCovariance innovation_covariance =
    jacobian * covariance_ * jacobian.transpose() + noise;
  Eigen::LDLT<MeasurementCovariance> decomposition(
    innovation_covariance);
  if (decomposition.info() != Eigen::Success
      || !decomposition.isPositive()) {
    return diagnostic;
  }

  // NIS 只作为基线诊断，不参与本轮关联和拒绝。
  const MeasurementVector solved_residual =
    decomposition.solve(residual);
  const double nis = residual.dot(solved_residual);
  diagnostic.innovation = residual;
  diagnostic.nis = nis;
  diagnostic.nis_valid = std::isfinite(nis) && nis >= 0.0;
  if (!diagnostic.nis_valid) {
    return diagnostic;
  }

  const Eigen::Matrix<double, STATE_DIM, kMeasurementDim> kalman_gain =
    decomposition.solve(jacobian * covariance_).transpose();
  StateVector corrected_state = x_ + kalman_gain * residual;
  corrected_state[YAW] = normalizeAngle(corrected_state[YAW]);

  // Joseph 形式避免简单协方差更新破坏对称性。
  const StateCovariance identity = StateCovariance::Identity();
  const StateCovariance correction = identity - kalman_gain * jacobian;
  StateCovariance corrected_covariance =
    correction * covariance_ * correction.transpose()
    + kalman_gain * noise * kalman_gain.transpose();
  corrected_covariance =
    0.5 * (corrected_covariance + corrected_covariance.transpose());
  applyModelConstraints(corrected_state, corrected_covariance);
  if (!validateState(corrected_state, corrected_covariance)) {
    return diagnostic;
  }

  x_ = corrected_state;
  covariance_ = corrected_covariance;
  diagnostic.accepted = true;
  return diagnostic;
}

void EkfTracker::finishFrame(bool found)
{
  // 帧末统一推进生命周期：命中推进确认计数，漏检按状态降级/重置。
  if (found) {
    if (tracker_state_ == TrackerState::Detecting) {
      ++successful_frame_count_;
      if (successful_frame_count_ >= config_.confirmation_hits) {
        tracker_state_ = TrackerState::Tracking;
      }
    } else if (tracker_state_ == TrackerState::TemporaryLost) {
      tracker_state_ = TrackerState::Tracking;
    }
    return;
  }

  if (tracker_state_ == TrackerState::Detecting) {
    initialized_ = false;
    tracker_state_ = TrackerState::Lost;
    successful_frame_count_ = 0;
  } else if (tracker_state_ == TrackerState::Tracking) {
    tracker_state_ = TrackerState::TemporaryLost;
  }
}

std::vector<AssociationDiagnostic> EkfTracker::update(
  const std::vector<ArmorObservation>& observations)
{
  // 帧内更新流程：收集本帧有效观测的最新时间戳 → 预测到该时刻 →
  // 未初始化则用最佳观测建立新目标 → 关联并逐个校正 → 帧末统一更新
  // 生命周期 → 汇总质量诊断并发布状态。
  const TrackerState lifecycle_before = tracker_state_;
  std::vector<std::size_t> valid_indices;
  TimePoint update_time{};
  bool has_update_time = false;
  for (std::size_t index = 0; index < observations.size(); ++index) {
    if (!validObservation(observations[index], robot_id_, model_)) {
      continue;
    }
    if (!has_update_time || observations[index].timestamp > update_time) {
      update_time = observations[index].timestamp;
      has_update_time = true;
    }
  }

  if (has_update_time && initialized_ && update_time > filter_time_) {
    predict(update_time);
  }
  // 只处理当前滤波时刻的观测，避免旧帧倒灌并污染新状态。
  const bool update_time_is_valid =
    has_update_time
    && (!initialized_ || update_time >= filter_time_);
  if (update_time_is_valid) {
    for (std::size_t index = 0; index < observations.size(); ++index) {
      if (observations[index].timestamp == update_time
          && validObservation(observations[index], robot_id_, model_)) {
        valid_indices.push_back(index);
      }
    }
  }

  std::vector<AssociationDiagnostic> diagnostics;
  std::vector<const ArmorObservation*> accepted_observations;
  bool initialized_this_frame = false;
  std::size_t anchor_index = std::numeric_limits<std::size_t>::max();
  // 首次观测初始化：选置信度最高、重投影误差最小的观测作为 0 号面锚点。
  if (!initialized_ && !valid_indices.empty()) {
    anchor_index = *std::max_element(
      valid_indices.begin(),
      valid_indices.end(),
      [&observations](std::size_t lhs, std::size_t rhs) {
        const auto& left = observations[lhs];
        const auto& right = observations[rhs];
        if (left.confidence != right.confidence) {
          return left.confidence < right.confidence;
        }
        return left.optimized_reprojection_error_px
               > right.optimized_reprojection_error_px;
      });
    initialize(observations[anchor_index]);
    initialized_this_frame = true;
    accepted_observations.push_back(&observations[anchor_index]);
    diagnostics.push_back({
      .observation_index = anchor_index,
      .robot_id = robot_id_,
      .associated_face_id = 0,
      .implied_radius_m = config_.initial_radius,
      .radius_error_m = 0.0,
      .minimum_corner_angle_rad =
        model_ == TargetModel::FourArmorVehicle
          ? std::numbers::pi / 2.0
          : 0.0,
      .innovation = Eigen::Vector4d::Zero(),
      .nis = 0.0,
      .nis_valid = true,
      .accepted = true,
      .lifecycle_before = lifecycle_before,
      .lifecycle_after = tracker_state_});
  }

  std::vector<std::size_t> association_indices;
  for (const std::size_t index : valid_indices) {
    if (index != anchor_index) {
      association_indices.push_back(index);
    }
  }
  for (const Association& association :
       associateObservations(observations, association_indices)) {
    AssociationDiagnostic diagnostic =
      correct(observations[association.observation_index], association);
    if (diagnostic.accepted) {
      accepted_observations.push_back(
        &observations[association.observation_index]);
    }
    diagnostics.push_back(std::move(diagnostic));
  }

  const bool found = !accepted_observations.empty();
  if (found) {
    last_seen_ = has_update_time ? update_time : filter_time_;
    updated_this_frame_ = true;
  } else {
    updated_this_frame_ = false;
  }

  // 初始化已经计为第一次命中，不能在同一帧重复增加确认计数。
  if (!initialized_this_frame) {
    finishFrame(found);
  }

  last_quality_ = {};
  if (found) {
    double confidence_sum = 0.0;
    double reprojection_sum = 0.0;
    for (const ArmorObservation* observation : accepted_observations) {
      confidence_sum += observation->confidence;
      reprojection_sum +=
        observation->optimized_reprojection_error_px;
    }
    last_quality_.accepted_observation_count =
      accepted_observations.size();
    last_quality_.mean_detection_confidence = static_cast<float>(
      confidence_sum
      / static_cast<double>(accepted_observations.size()));
    last_quality_.mean_reprojection_error_px =
      reprojection_sum
      / static_cast<double>(accepted_observations.size());
    const auto accepted = std::find_if(
      diagnostics.rbegin(),
      diagnostics.rend(),
      [](const auto& diagnostic) {
        return diagnostic.accepted;
      });
    if (accepted != diagnostics.rend()) {
      last_quality_.last_nis = accepted->nis;
      last_quality_.nis_valid = accepted->nis_valid;
      last_quality_.associated_face_id =
        accepted->associated_face_id;
      last_quality_.innovation = accepted->innovation;
    }
  }
  last_quality_.lifecycle_before = lifecycle_before;
  last_quality_.lifecycle_after = tracker_state_;
  for (auto& diagnostic : diagnostics) {
    diagnostic.lifecycle_before = lifecycle_before;
    diagnostic.lifecycle_after = tracker_state_;
  }

  if (!found && lifecycle_before != tracker_state_) {
    diagnostics.push_back({
      .robot_id = robot_id_,
      .lifecycle_before = lifecycle_before,
      .lifecycle_after = tracker_state_});
  }
  publishState();
  return diagnostics;
}

void EkfTracker::applyModelConstraints(
  StateVector& state,
  StateCovariance& covariance) const noexcept
{
  // 前哨站锁零 radius_offset 与 height_offset。
  if (model_ != TargetModel::ThreeArmorOutpost) {
    return;
  }

  state[RADIUS_OFFSET] = 0.0;
  state[HEIGHT_OFFSET] = 0.0;
  covariance.row(RADIUS_OFFSET).setZero();
  covariance.col(RADIUS_OFFSET).setZero();
  covariance.row(HEIGHT_OFFSET).setZero();
  covariance.col(HEIGHT_OFFSET).setZero();
}

bool EkfTracker::validateState(
  const StateVector& state,
  const StateCovariance& covariance) const
{
  // 有限性、半径/几何/高度约束与协方差半正定检查。
  if (!state.allFinite() || !covariance.allFinite()) {
    return false;
  }

  const double first_radius = state[RADIUS];
  if (first_radius < config_.min_radius
      || first_radius > config_.max_radius) {
    return false;
  }
  if (model_ == TargetModel::FourArmorVehicle) {
    const double second_radius =
      state[RADIUS] + state[RADIUS_OFFSET];
    if (second_radius < config_.min_radius
        || second_radius > config_.max_radius
        || (config_.enable_vehicle_geometry_constraints
            && fourArmorMinimumCornerAngle(first_radius, second_radius)
                 + kGeometryTolerance
                 < config_.minimum_four_armor_corner_angle_rad)
        || std::abs(state[HEIGHT_OFFSET])
             > config_.max_abs_height_offset) {
      return false;
    }
  } else if (std::abs(state[RADIUS_OFFSET]) > 1e-12
             || std::abs(state[HEIGHT_OFFSET]) > 1e-12) {
    return false;
  }

  const StateCovariance symmetric =
    0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<StateCovariance> eigen_solver(
    symmetric,
    Eigen::EigenvaluesOnly);
  return eigen_solver.info() == Eigen::Success
         && eigen_solver.eigenvalues().minCoeff()
              >= -kCovarianceTolerance;
}

void EkfTracker::publishState()
{
  state_ = TargetState{};
  state_.robot_id = robot_id_;
  state_.model = model_;
  state_.armor_count = armorFaceCount();
  state_.tracker_state = tracker_state_;
  state_.last_observation_time = last_seen_;
  state_.updated_this_frame = updated_this_frame_;
  state_.quality = last_quality_;
  // Detecting 阶段不发布数值状态（state 保持默认值，yaw 为 0），
  // 调用方不应在未确认前使用 state() 的位置/yaw。
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
  state_.radius_offset =
    model_ == TargetModel::FourArmorVehicle
      ? x_[RADIUS_OFFSET]
      : 0.0;
  state_.height_offset =
    model_ == TargetModel::FourArmorVehicle
      ? x_[HEIGHT_OFFSET]
      : 0.0;
  state_.covariance = covariance_;
  state_.timestamp = filter_time_;
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
  // 超过过期时间或从未初始化即视为过期。
  if (!initialized_ || tracker_state_ == TrackerState::Lost
      || now < last_seen_) {
    return true;
  }
  return now - last_seen_ > config_.expiration_timeout;
}

bool EkfTracker::initialized() const noexcept
{
  return initialized_;
}

}  // namespace L3Estimation
