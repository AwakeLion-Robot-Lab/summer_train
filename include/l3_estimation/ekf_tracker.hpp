#pragma once

#include "l3_estimation/types.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cstddef>
#include <numbers>
#include <vector>

namespace L3Estimation {

// EKF、关联和生命周期的可调参数。
struct EkfTrackerConfig {
  bool enable_vehicle_geometry_constraints = true;
  double initial_radius = 0.20;
  // 11 维状态初始方差。
  StateVector initial_variance =
    (StateVector{} << 1.0, 64.0, 1.0, 64.0, 1.0, 64.0,
     0.4, 100.0, 0.01, 0.01, 0.01).finished();

  // 位置与 yaw 的分段白噪声加速度方差。
  double linear_acceleration_variance = 100.0;
  double angular_acceleration_variance = 400.0;
  // 几何量（半径/高度差）的随机游走方差。
  double geometry_random_walk_variance = 1e-5;

  // 观测噪声：位置基 std + 距离二次项，以及 yaw std。
  double position_standard_deviation_base_m = 0.05;
  double position_standard_deviation_quadratic = 0.01;
  double armor_yaw_standard_deviation_rad = 0.15;

  // 生命周期：确认帧数、最大预测间隔、过期时间。
  int confirmation_hits = 5;
  std::chrono::milliseconds max_predict_interval{100};
  std::chrono::milliseconds expiration_timeout{500};

  // 关联门限、权重与 NIS 参考阈值。
  double association_position_gate = 0.60;
  double association_yaw_gate = std::numbers::pi / 3.0;
  double association_radius_gate = 0.20;
  double association_position_weight = 1.0;
  double association_yaw_weight = 1.0;
  double association_radius_weight = 0.5;
  double nis_reference_threshold = 9.4877;

  // 整车几何安全范围：半径区间、最小内角和最大高度差。
  double min_radius = 0.05;
  double max_radius = 0.50;
  double minimum_four_armor_corner_angle_rad =
    50.0 * std::numbers::pi / 180.0;
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
    // 观测-物理面候选：门限检查后的误差与归一化代价。
    std::size_t observation_index = 0;
    int face_id = -1;
    double position_error_m = 0.0;
    double yaw_error_rad = 0.0;
    double implied_radius_m = 0.0;
    double radius_error_m = 0.0;
    double minimum_corner_angle_rad = 0.0;
    double match_cost = 0.0;
  };

  // 用首块观测建立新目标（0 号面锚点）。
  void initialize(const ArmorObservation& observation);
  // 帧末统一推进生命周期。
  void finishFrame(bool found);

  // 模型面数与第 face_id 面相对 yaw。
  [[nodiscard]] int armorFaceCount() const noexcept;
  [[nodiscard]] double faceAngle(int face_id) const noexcept;
  // 奇数面是否使用第二组半径/高度（仅四板车）。
  [[nodiscard]] bool usesSecondGeometryGroup(int face_id) const noexcept;

  // 匀速 + 匀角速度的状态转移矩阵与过程噪声。
  [[nodiscard]] StateCovariance buildTransition(double dt) const noexcept;
  [[nodiscard]] StateCovariance buildProcessNoise(double dt) const noexcept;

  // 第 face_id 面装甲中心的预测位置。
  [[nodiscard]] Eigen::Vector3d armorPosition(
    const StateVector& state,
    int face_id) const noexcept;

  // 面中心位置 + 面 yaw 的预测观测。
  [[nodiscard]] MeasurementVector predictMeasurement(
    const StateVector& state,
    int face_id) const;

  // 观测函数对 11 维状态的分析 Jacobian。
  [[nodiscard]] MeasurementJacobian measurementJacobian(
    const StateVector& state,
    int face_id) const;

  // 距离相关的观测噪声（按置信度缩放）。
  [[nodiscard]] MeasurementCovariance measurementNoise(
    const ArmorObservation& observation) const noexcept;

  // 枚举观测×物理面候选并做贪心一对一选择。
  [[nodiscard]] std::vector<Association> associateObservations(
    const std::vector<ArmorObservation>& observations,
    const std::vector<std::size_t>& observation_indices) const;

  // 单块观测的 EKF 校正（Joseph 形式，NIS 只记录）。
  [[nodiscard]] AssociationDiagnostic correct(
    const ArmorObservation& observation,
    const Association& association);

  // 状态与协方差合法性（有限性、几何约束、半正定）。
  [[nodiscard]] bool validateState(
    const StateVector& state,
    const StateCovariance& covariance) const;

  // 前哨站锁零两个 offset 对应的状态与协方差行列。
  void applyModelConstraints(
    StateVector& state,
    StateCovariance& covariance) const noexcept;

  // 把内部状态写入对外 TargetState。
  void publishState();

  // 观测与预测的 [x,y,z,yaw] 向量及其角度归一化残差。
  [[nodiscard]] static MeasurementVector observationMeasurement(
    const ArmorObservation& observation) noexcept;
  [[nodiscard]] static MeasurementVector measurementResidual(
    const MeasurementVector& measured,
    const MeasurementVector& predicted) noexcept;

  // 本 Tracker 归属的 robot 与模型。
  int robot_id_ = -1;
  TargetModel model_ = TargetModel::FourArmorVehicle;
  EkfTrackerConfig config_;

  // 滤波状态/协方差与对外状态。
  StateVector x_ = StateVector::Zero();
  StateCovariance covariance_ = StateCovariance::Identity();
  TargetState state_{};

  // 生命周期：当前状态、滤波时间、最近观测时间、连续命中数。
  TrackerState tracker_state_ = TrackerState::Lost;
  TimePoint filter_time_{};
  TimePoint last_seen_{};
  int successful_frame_count_ = 0;
  bool initialized_ = false;
  bool updated_this_frame_ = false;
  TargetQualityMetrics last_quality_{};
};

}  // namespace L3Estimation
