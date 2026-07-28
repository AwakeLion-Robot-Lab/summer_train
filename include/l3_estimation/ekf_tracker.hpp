#pragma once

#include "l3_estimation/types.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cstddef>
#include <numbers>
#include <vector>

namespace L3Estimation {

struct EkfTrackerConfig {
  double initial_radius = 0.20;
  StateVector initial_variance =
    (StateVector{} << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0,
     0.4, 100.0, 0.01, 0.01, 0.01).finished();

  double linear_acceleration_variance = 100.0;
  double angular_acceleration_variance = 400.0;
  double geometry_random_walk_variance = 1e-5;

  double position_standard_deviation_base_m = 0.05;
  double position_standard_deviation_quadratic = 0.01;
  double armor_yaw_standard_deviation_rad = 0.15;

  int confirmation_hits = 5;
  std::chrono::milliseconds max_predict_interval{100};
  std::chrono::milliseconds expiration_timeout{500};

  double association_position_gate = 0.60;
  double association_yaw_gate = std::numbers::pi / 3.0;
  double association_position_weight = 1.0;
  double association_yaw_weight = 1.0;
  double nis_reference_threshold = 9.4877;

  double min_radius = 0.05;
  double max_radius = 0.50;
  double max_abs_height_offset = 0.30;
};

// 一个 EkfTracker 只维护一个 robot_id 的一个整车假设。
class EkfTracker {
public:
  explicit EkfTracker(
    int robot_id = -1,
    EkfTrackerConfig config = {});

  EkfTracker(
    int robot_id,
    TargetModel model,
    EkfTrackerConfig config = {});

  // predict 只推进数值状态，生命周期在本帧 update 结束后统一改变。
  void predict(TimePoint timestamp);

  [[nodiscard]] std::vector<AssociationDiagnostic> update(
    const std::vector<ArmorObservation>& observations);

  void reset();

  [[nodiscard]] const TargetState& state() const noexcept;
  [[nodiscard]] TrackerState trackerState() const noexcept;
  [[nodiscard]] bool expired(TimePoint now) const noexcept;
  [[nodiscard]] bool initialized() const noexcept;

private:
  static constexpr int kMeasurementDim = 4;

  using MeasurementVector = Eigen::Vector4d;
  using MeasurementJacobian =
    Eigen::Matrix<double, kMeasurementDim, STATE_DIM>;
  using MeasurementCovariance = Eigen::Matrix4d;

  struct Association {
    std::size_t observation_index = 0;
    int face_id = -1;
    double position_error_m = 0.0;
    double yaw_error_rad = 0.0;
    double match_cost = 0.0;
  };

  void initialize(const ArmorObservation& observation);
  void finishFrame(bool found);

  [[nodiscard]] int armorFaceCount() const noexcept;
  [[nodiscard]] double faceAngle(int face_id) const noexcept;
  [[nodiscard]] bool usesSecondGeometryGroup(int face_id) const noexcept;

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
    const std::vector<std::size_t>& observation_indices) const;

  [[nodiscard]] AssociationDiagnostic correct(
    const ArmorObservation& observation,
    const Association& association);

  [[nodiscard]] bool validateState(
    const StateVector& state,
    const StateCovariance& covariance) const;

  void applyModelConstraints(
    StateVector& state,
    StateCovariance& covariance) const noexcept;

  void publishState();

  [[nodiscard]] static double normalizeAngle(double angle) noexcept;
  [[nodiscard]] static MeasurementVector observationMeasurement(
    const ArmorObservation& observation) noexcept;
  [[nodiscard]] static MeasurementVector measurementResidual(
    const MeasurementVector& measured,
    const MeasurementVector& predicted) noexcept;

  int robot_id_ = -1;
  TargetModel model_ = TargetModel::FourArmorVehicle;
  EkfTrackerConfig config_;

  StateVector x_ = StateVector::Zero();
  StateCovariance covariance_ = StateCovariance::Identity();
  TargetState state_{};

  TrackerState tracker_state_ = TrackerState::Lost;
  TimePoint filter_time_{};
  TimePoint last_seen_{};
  int successful_frame_count_ = 0;
  bool initialized_ = false;
  bool updated_this_frame_ = false;
  TargetQualityMetrics last_quality_{};
};

}  // namespace L3Estimation
