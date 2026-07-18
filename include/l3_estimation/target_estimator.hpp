#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/ekf_tracker.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Geometry>

#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace L3Estimation {

using GimbalPoseProvider =
  std::function<std::optional<Eigen::Quaterniond>(TimePoint)>;

// L3 的唯一对外入口：二维装甲板检测 → 可供 L4 使用的整车状态。
class TargetEstimator {
public:
  TargetEstimator(
    L1Sensor::CameraCalibration calibration,
    Eigen::Matrix3d rotation_camera_to_gimbal,
    Eigen::Vector3d translation_camera_to_gimbal,
    GimbalPoseProvider gimbal_pose_provider);

  [[nodiscard]] std::vector<VehicleState> update(
    const std::vector<L2Perception::ArmorDetection>& armors,
    TimePoint timestamp);

private:
  [[nodiscard]] ArmorSize armorSizeFromClass(int class_id) const noexcept;

  [[nodiscard]] std::optional<ArmorObservation> makeObservation(
    const L2Perception::ArmorDetection& armor,
    TimePoint timestamp) const;

  PnpSolver pnp_solver_;
  Eigen::Matrix3d rotation_camera_to_gimbal_;
  Eigen::Vector3d translation_camera_to_gimbal_;
  GimbalPoseProvider gimbal_pose_provider_;
  std::unordered_map<int, EkfTracker> trackers_;
};

}  // namespace L3Estimation
