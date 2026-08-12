#pragma once

#ifdef NEWVISION_USE_GTSAM

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/gtsam_est/config.hpp"
#include "l3_estimation/gtsam_est/factors.hpp"
#include "l3_estimation/tracked_target.hpp"
#include "l3_estimation/types.hpp"

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace L3Estimation::GtsamEst {

// GTSAM 图变量只在这个命名空间出现。半径 A/B 是 logistic 变换之前的无界变量；
// deltaZ 是物理量。每帧的单板 Pose3 使用 armorPose(k, id)。
namespace keys {
[[nodiscard]] gtsam::Key center(std::uint64_t k) noexcept;
[[nodiscard]] gtsam::Key velocity(std::uint64_t k) noexcept;
[[nodiscard]] gtsam::Key yaw(std::uint64_t k) noexcept;
[[nodiscard]] gtsam::Key vyaw(std::uint64_t k) noexcept;
[[nodiscard]] gtsam::Key radiusA() noexcept;
[[nodiscard]] gtsam::Key radiusB() noexcept;
[[nodiscard]] gtsam::Key deltaZ() noexcept;
[[nodiscard]] gtsam::Key armorPose(std::uint64_t k, int armor_id);
}  // namespace keys

// GTSAM 后端自己的目标。它不包含 ExtendedKalmanFilter，也不借用 FilterEst 的
// 目标类型；snapshot() 是它与公共层唯一的数据转换边界。
class Target
{
public:
  Target(
    TargetConfig target_config,
    Config gtsam_config,
    ArmorConfig armor_config,
    const L1Sensor::CameraCalibration& calibration);

  void initialize(
    const Armor& armor,
    TimePoint timestamp,
    const Eigen::Isometry3d& T_world_camera);

  // 无匹配观测时仍添加运动状态，保持 TempLost 期间的恒速度/恒角速度预测。
  [[nodiscard]] bool update(
    const std::vector<const Armor*>& armors,
    TimePoint timestamp,
    const Eigen::Isometry3d& T_world_camera);

  [[nodiscard]] TrackedTarget snapshot() const;
  [[nodiscard]] std::vector<Eigen::Vector4d> armorPoses() const;
  [[nodiscard]] bool diverged() const noexcept;
  [[nodiscard]] ArmorName name() const noexcept { return name_; }
  [[nodiscard]] ArmorType armorType() const noexcept { return armor_type_; }
  [[nodiscard]] int armorCount() const noexcept { return armor_count_; }
  [[nodiscard]] std::size_t activeVariableCount() const noexcept;

private:
  [[nodiscard]] std::optional<int> matchArmor(
    const Armor& armor,
    const std::vector<bool>& used_ids) const;
  void initializeStateFromArmor(const Armor& armor);
  void addMotionState(std::uint64_t k, double dt);
  void addStaticGeometry(double initial_radius);
  void addArmorFactors(
    std::uint64_t k,
    const Armor& armor,
    int armor_id,
    const Eigen::Isometry3d& T_world_camera);
  void commitPendingGraph();
  void readEstimate();

  TargetConfig target_config_{};
  Config gtsam_config_{};
  ArmorConfig armor_config_{};
  L1Sensor::CameraCalibration calibration_{};

  ArmorName name_{ArmorName::Unknown};
  ArmorType armor_type_{ArmorType::Small};
  int armor_count_{0};
  int last_id_{0};
  bool failed_{false};
  bool isam_started_{false};

  std::uint64_t k_{0};
  TimePoint timestamp_{};
  TimePoint last_observation_timestamp_{};
  TargetStateVector state_{TargetStateVector::Zero()};
  TargetCovariance covariance_{TargetCovariance::Identity()};

  // 与 JLU RobotTarget 一致：目标 LOST 时整个 Target 销毁，新目标从空 ISAM2 开始。
  gtsam::ISAM2 isam2_;
  gtsam::Values pending_values_;
  gtsam::NonlinearFactorGraph pending_graph_;
};

}  // namespace L3Estimation::GtsamEst

#endif  // NEWVISION_USE_GTSAM
