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
  // 类别 → 尺寸 / robot_id / 目标模型；基地与未知类别不生成观测。
  [[nodiscard]] ArmorSize armorSizeFromClass(int class_id) const noexcept;
  [[nodiscard]] int robotIdFromClass(int class_id) const noexcept;
  [[nodiscard]] std::optional<TargetModel> targetModelFromClass(
    int class_id) const noexcept;

  // 逐检测生成世界系观测。
  [[nodiscard]] std::vector<ArmorObservation> buildObservations(
    const std::vector<L2Perception::ArmorDetection>& armors,
    TimePoint timestamp,
    const Eigen::Quaterniond& R_world_barrel) const;

  [[nodiscard]] std::optional<ArmorObservation> makeObservation(
    const L2Perception::ArmorDetection& armor,
    std::size_t source_detection_index,
    TimePoint timestamp,
    const Eigen::Quaterniond& R_world_barrel) const;

  [[nodiscard]] std::optional<ArmorObservation> buildObservation(
    const L2Perception::ArmorDetection& armor,
    std::size_t source_detection_index,
    TimePoint timestamp,
    const Eigen::Quaterniond& R_world_barrel,
    int robot_id,
    TargetModel model,
    const YawOptimizationResult& yaw) const;

  // 已确认目标：按 raw yaw 与最近预测 face yaw 的角距离选候选（平手依次比较
  // raw 误差、候选下标）；返回最优候选并把其角距离写入 best_distance_out。
  [[nodiscard]] const ArmorPose* selectCandidateByPredictedFace(
    const L2Perception::ArmorDetection& armor,
    ArmorSize armor_size,
    const std::vector<ArmorPose>& poses,
    const Eigen::Quaterniond& R_world_barrel,
    double configured_pitch,
    const std::vector<double>& predicted_faces,
    double& best_distance_out) const;

  // 回退路径：对每个候选做 yaw 搜索，按优化重投影误差选解（平手依次比较
  // raw 误差、候选下标）。
  [[nodiscard]] std::optional<YawOptimizationResult>
  selectCandidateByOptimizedError(
    const L2Perception::ArmorDetection& armor,
    ArmorSize armor_size,
    const std::vector<ArmorPose>& poses,
    const Eigen::Quaterniond& R_world_barrel,
    double configured_pitch) const;

  // 已确认目标（Tracking/TemporaryLost）预测到当前帧时，返回该车的
  // 各物理面预测 yaw；否则返回 nullopt。
  [[nodiscard]] std::optional<std::vector<double>> predictedFaceYaws(
    int robot_id,
    TargetModel model,
    TimePoint timestamp) const;

  // 相机系 tvec → barrel → world。
  [[nodiscard]] Eigen::Vector3d positionInWorld(
    const cv::Vec3d& tvec_camera,
    const Eigen::Quaterniond& R_world_barrel) const noexcept;

  // 所有 Tracker 推进到帧时间戳。
  void predictTrackers(TimePoint timestamp);
  // 按 robot_id 分组更新已有 Tracker，并为新 robot 建立 Tracker。
  void updateTrackers(const std::vector<ArmorObservation>& observations);
  // 删除超过过期时间的 Tracker。
  void removeExpiredTrackers(TimePoint timestamp);
  // 汇总并排序对外发布的 TargetState。
  [[nodiscard]] std::vector<TargetState> collectTargets() const;

  // 配置与依赖组件。
  L3Config config_;
  PnpSolver pnp_solver_;
  YawOptimizer yaw_optimizer_;
  Eigen::Isometry3d T_barrel_camera_ = Eigen::Isometry3d::Identity();
  cv::Size calibration_image_size_{};
  BarrelPoseProvider barrel_pose_provider_;
  std::unordered_map<int, EkfTracker> trackers_;
  // 每帧观测与关联诊断缓存，供回放/调试读取。
  std::vector<ArmorObservation> last_observations_;
  std::vector<AssociationDiagnostic> last_association_diagnostics_;
};

}  // namespace L3Estimation
