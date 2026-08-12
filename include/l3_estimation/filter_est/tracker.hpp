#pragma once

#include "l2_perception/armor.hpp"
#include "l3_estimation/filter_est/target.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/tracked_target.hpp"
#include "l3_estimation/tracker.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Geometry>

#include <optional>
#include <vector>

// 滤波后端：整车扩展卡尔曼滤波。这是本项目的默认估计器，也是当前唯一经过回放
// 验证的一条路径。因子图后端见 l3_estimation/gtsam_est/。
namespace L3Estimation::FilterEst {

// 逐帧递推的整车 EKF 跟踪器：一帧一次 predict + 若干次 update，只保留当前时刻的
// 状态和协方差，不回头修正历史。代价是每帧 O(1)，但装甲板关联一旦错了就没有
// 机会再纠正——这正是因子图后端想改进的地方。
class Tracker : public ITracker {
public:
  // 相机标定无效时对象仍可构造，但 ready() 为 false，track() 只返回空。
  explicit Tracker(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig armor_config = {},
    TrackerConfig tracker_config = {},
    L3Estimation::TargetConfig target_config = {},
    TargetConfig filter_config = {});

  [[nodiscard]] bool ready() const noexcept override;
  [[nodiscard]] TrackState state() const noexcept override;
  [[nodiscard]] EstimatorBackend backend() const noexcept override;

  // 处理一帧检测。枪管姿态必须对应 timestamp 所表示的图像曝光时刻。
  // Lost 或初始化失败时返回空；TempLost 时返回纯预测状态。
  [[nodiscard]] std::optional<TrackedTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp) override;

  // 本帧全部 PnP 结果（含被丢弃的），只读调试快照。
  [[nodiscard]] const std::vector<Armor>& observations() const noexcept override;

  // 当前 EKF 整车模型展开的装甲板 [x, y, z, yaw]，Lost 时为空。
  [[nodiscard]] std::vector<Eigen::Vector4d> targetArmorPoses() const override;

  // 清空状态机、时间连续性记录和本帧观测。
  void reset() noexcept override;

private:
  // 挑出 PnP 成功的观测，并按到图像中心的距离排序。
  [[nodiscard]] std::vector<const Armor*> usableObservations() const;
  // 最近的 NIS 失败比例是否已经超过门限。
  [[nodiscard]] bool nisPersistentlyBad(const Target& target) const;
  // Lost 状态下建立新目标，或对已有目标做预测 + 观测更新。
  [[nodiscard]] bool initializeTarget(
    const std::vector<const Armor*>& observations,
    TimePoint timestamp);
  [[nodiscard]] bool updateTarget(
    const std::vector<const Armor*>& observations,
    TimePoint timestamp);
  // 根据本帧是否关联到观测推进四态状态机。
  void updateState(bool found);
  // 只重置跟踪生命周期；帧时间和诊断观测由调用方按需清理。
  void resetTracking() noexcept;

  ArmorConfig armor_config_;
  TrackerConfig tracker_config_;
  L3Estimation::TargetConfig target_config_;
  TargetConfig filter_config_;
  PnpSolver pnp_solver_;
  cv::Point2f image_center_{};
  bool ready_{false};

  // 状态机计数、当前目标、上一帧时间及本帧全部观测。
  TrackState state_{TrackState::Lost};
  int detect_count_{0};
  int temp_lost_count_{0};
  std::optional<Target> target_;
  std::optional<TimePoint> last_timestamp_;
  std::vector<Armor> observations_;
};

}  // namespace L3Estimation::FilterEst
