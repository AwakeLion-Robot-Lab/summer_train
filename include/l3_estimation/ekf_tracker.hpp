#pragma once

#include "l3_estimation/types.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cstddef>
#include <numbers>
#include <vector>

namespace L3Estimation {

// EkfTracker 的全部可调参数。默认值以 tongjiceshi 的车辆模型为起点，
// 后续 Runtime 可以从 YAML 读取后通过 TargetEstimator 统一传入。
struct EkfTrackerConfig {
  double initial_radius = 0.20;// 初始半径
  StateVector initial_variance =
    (StateVector{} << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0,
     0.4, 100.0, 0.01, 0.01, 0.01) 
      .finished();

  double linear_acceleration_variance = 100.0;
  double angular_acceleration_variance = 400.0;
  double geometry_random_walk_variance = 1e-5;

  int confirmation_hits = 5;
  std::chrono::milliseconds max_predict_interval{100};
  std::chrono::milliseconds expiration_timeout{300};

  double association_position_gate = 0.60;
  double association_yaw_gate = std::numbers::pi / 3.0;
  double nis_gate = 9.4877;

  double min_radius = 0.05;
  double max_radius = 0.50;
  double max_abs_height_offset = 0.30;
};

// 一个 EkfTracker 只维护一辆由 robot_id 标识的四装甲板车辆。
class EkfTracker {
public:
  explicit EkfTracker(
    int robot_id = -1,
    EkfTrackerConfig config = {});

  void predict(TimePoint timestamp);
  void update(const std::vector<ArmorObservation>& observations);
  void reset();

  [[nodiscard]] const TargetState& state() const noexcept;
  [[nodiscard]] TrackerState trackerState() const noexcept;
  [[nodiscard]] bool expired(TimePoint now) const noexcept;

private:
  static constexpr int kArmorFaceCount = 4;
  static constexpr int kMeasurementDim = 4;

  using MeasurementVector = Eigen::Matrix<double, kMeasurementDim, 1>;
  using MeasurementJacobian =
    Eigen::Matrix<double, kMeasurementDim, STATE_DIM>;
  using MeasurementCovariance =
    Eigen::Matrix<double, kMeasurementDim, kMeasurementDim>;

  struct Association {
    std::size_t observation_index = 0;
    int face_id = 0;
    double nis = 0.0;
  };

  void initialize(const ArmorObservation& observation);

  [[nodiscard]] StateCovariance buildTransition(double dt) const noexcept;
  [[nodiscard]] StateCovariance buildProcessNoise(double dt) const noexcept;

  [[nodiscard]] Eigen::Vector3d armorPosition(
    const StateVector& state,
    int face_id) const noexcept;

  [[nodiscard]] MeasurementVector predictMeasurement(
    const StateVector& state,
    int face_id) const;

  [[nodiscard]] MeasurementJacobian measurementJacobian(
    const StateVector& state,
    int face_id) const;

  [[nodiscard]] MeasurementCovariance measurementNoise(
    const ArmorObservation& observation) const noexcept;

  [[nodiscard]] std::vector<Association> associateObservations(
    const std::vector<ArmorObservation>& observations,
    const std::vector<std::size_t>& observation_indices,
    unsigned int reserved_face_mask = 0U) const;

  [[nodiscard]] bool correct(
    const ArmorObservation& observation,
    int face_id);

  [[nodiscard]] bool validateState(
    const StateVector& state,
    const StateCovariance& covariance) const;

  void publishState();

  [[nodiscard]] static double normalizeAngle(double angle) noexcept;
  [[nodiscard]] static MeasurementVector observationMeasurement(
    const ArmorObservation& observation) noexcept;
  [[nodiscard]] static MeasurementVector measurementResidual(
    const MeasurementVector& measured,
    const MeasurementVector& predicted) noexcept;

  int robot_id_ = -1;
  EkfTrackerConfig config_;

  StateVector x_ = StateVector::Zero();
  StateCovariance covariance_ = StateCovariance::Identity();
  TargetState state_{};

  TrackerState tracker_state_ = TrackerState::Lost;
  TimePoint filter_time_{};
  TimePoint last_seen_{};
  int successful_frame_count_ = 0;
  bool initialized_ = false;
  bool awaiting_update_ = false;
};

}  // namespace L3Estimation
