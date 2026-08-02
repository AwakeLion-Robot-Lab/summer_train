#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/config.hpp"
#include "l3_estimation/ekf_tracker.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/types.hpp"
#include "l3_estimation/yaw_optimizer.hpp"

#include <Eigen/Geometry>

#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace L3Estimation {

// 返回图像采样时刻的 R_world_barrel。
using BarrelPoseProvider =
  std::function<std::optional<Eigen::Quaterniond>(TimePoint)>;

// L3 唯一入口：二维装甲板检测转换为世界系整车状态。
class TargetEstimator {
public:
  TargetEstimator(
    L1Sensor::CameraCalibration calibration,
    BarrelPoseProvider barrel_pose_provider,
    L3Config config = {});

  [[nodiscard]] std::vector<TargetState> update(
    const std::vector<L2Perception::ArmorDetection>& armors,
    const FrameContext& frame_context);

  [[nodiscard]] const std::vector<ArmorObservation>& lastObservations()
    const noexcept;

  [[nodiscard]] const std::vector<AssociationDiagnostic>&
  lastAssociationDiagnostics() const noexcept;

private:
  [[nodiscard]] ArmorSize armorSizeFromClass(int class_id) const noexcept;
  [[nodiscard]] int robotIdFromClass(int class_id) const noexcept;
  [[nodiscard]] std::optional<TargetModel> targetModelFromClass(
    int class_id) const noexcept;

  [[nodiscard]] std::vector<ArmorObservation> buildObservations(
    const std::vector<L2Perception::ArmorDetection>& armors,
    TimePoint timestamp,
    const Eigen::Quaterniond& R_world_barrel) const;

  [[nodiscard]] std::optional<ArmorObservation> makeObservation(
    const L2Perception::ArmorDetection& armor,
    TimePoint timestamp,
    const Eigen::Quaterniond& R_world_barrel) const;

  [[nodiscard]] Eigen::Vector3d positionInWorld(
    const ArmorPose& pose,
    const Eigen::Quaterniond& R_world_barrel) const noexcept;

  void predictTrackers(TimePoint timestamp);
  void updateTrackers(const std::vector<ArmorObservation>& observations);
  void removeExpiredTrackers(TimePoint timestamp);
  [[nodiscard]] std::vector<TargetState> collectTargets() const;

  L3Config config_;
  PnpSolver pnp_solver_;
  YawOptimizer yaw_optimizer_;
  Eigen::Isometry3d T_barrel_camera_ = Eigen::Isometry3d::Identity();
  cv::Size calibration_image_size_{};
  BarrelPoseProvider barrel_pose_provider_;
  std::unordered_map<int, EkfTracker> trackers_;
  std::vector<ArmorObservation> last_observations_;
  std::vector<AssociationDiagnostic> last_association_diagnostics_;
};

}  // namespace L3Estimation
