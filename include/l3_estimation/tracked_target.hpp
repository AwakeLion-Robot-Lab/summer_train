#pragma once

#include "l3_estimation/target_state.hpp"
#include "l3_estimation/types.hpp"

#include <limits>
#include <vector>

namespace L3Estimation {

// L3 -> L4 的后端中立估计快照。
//
// 它只保存公共物理状态、协方差和目标元数据；不拥有 ExtendedKalmanFilter、
// gtsam::Values 或任何后端生命周期。FilterEst::Target 与 GtsamEst::Target 各自
// 维护内部状态，只有在 ITracker::track() 返回时才转换成这个值类型。
class TrackedTarget
{
public:
  ArmorName name{ArmorName::Unknown};
  ArmorType armor_type{ArmorType::Small};
  bool jumped{false};
  int last_id{0};
  // 后端可选的一致性统计量。EKF 填 NIS；没有等价统计量的后端保持 NaN。
  double normalized_innovation_squared{
    std::numeric_limits<double>::quiet_NaN()};

  TrackedTarget() = default;
  TrackedTarget(
    ArmorName name,
    ArmorType armor_type,
    int armor_count,
    TimePoint timestamp,
    const TargetStateVector& state,
    const TargetCovariance& covariance,
    TargetConfig config = {});

  // 无观测的确定性构造入口，仅用于规划离线测试。
  TrackedTarget(double x, double vyaw, double radius, double height, double yaw = 0.0);

  [[nodiscard]] const TargetStateVector& state() const noexcept { return state_; }
  [[nodiscard]] const TargetCovariance& covariance() const noexcept { return covariance_; }
  [[nodiscard]] int armorCount() const noexcept { return armor_count_; }
  [[nodiscard]] TimePoint timestamp() const noexcept { return timestamp_; }
  [[nodiscard]] bool valid() const noexcept;

  // 只对快照做后端中立的运动学外推，供 L4 延迟补偿使用。
  void predict(TimePoint timestamp);
  void predict(double dt);

  [[nodiscard]] std::vector<Eigen::Vector4d> armorPoses() const;

private:
  // 共享物理运动模型参数，不属于 EKF 或 GTSAM 的内部数据结构。
  TargetConfig config_{};
  int armor_count_{0};
  TimePoint timestamp_{};
  TargetStateVector state_{TargetStateVector::Zero()};
  TargetCovariance covariance_{TargetCovariance::Zero()};
};

}  // namespace L3Estimation
