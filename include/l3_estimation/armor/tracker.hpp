#pragma once

#include "l2_perception/armor.hpp"
#include "l3_estimation/armor/pnp_solver.hpp"
#include "l3_estimation/armor/target_estimator.hpp"
#include "l3_estimation/armor/types.hpp"

#include <Eigen/Geometry>

#include <cstdint>
#include <optional>
#include <vector>

namespace L3Estimation {

// L2 -> L3 的显式转换入口。这里只复制检测字段，位姿字段由 Tracker
// 内部持有的 PnpSolver 补充。
Armor toArmorObservation(
  const L2Perception::Armor& detection,
  TimePoint timestamp);

// 维护单辆车的状态机和整车 EKF。每帧先完成 L2 -> L3 观测构造，再进行目标
// 初始化或更新，最后把当前 TrackedTarget 的副本交给下游。下游在自己的副本上
// 外推，不会影响滤波器状态。
class Tracker {
public:
  // 相机标定、装甲板配置或状态机配置无效时 ready() 返回 false。
  Tracker(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig armor_config = {},
    TrackerConfig tracker_config = {},
    TargetConfig target_config = {});

  bool ready() const noexcept;
  TrackState state() const noexcept;

  // 更新相机模型；坐标原点和世界系约定必须保持一致。仿真相机的安装位置
  // 可能随视角切换变化，不能一直使用启动时的外参。有效更新保留跟踪历史。
  bool setCalibration(const L1Sensor::CameraCalibration& calibration);

  // 连续确认计数，达到 min_detect_count 才从 Detecting 升到 Tracking。
  int detectCount() const noexcept { return detect_count_; }

  // resetTracking() 的累计次数。瞬时的 TrackState 看不出"每隔几帧重建一次
  // EKF"这种churn——状态图上只是一个单帧尖峰，很容易被看漏；单调计数器不会。
  std::uint64_t resetCount() const noexcept { return reset_count_; }

  // 处理一帧检测。枪管姿态必须对应 timestamp 所表示的图像曝光时刻。
  // Lost 或初始化失败时返回空；TempLost 时返回纯预测状态。
  [[nodiscard]] std::optional<TrackedTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp);

  // 保留本帧所有 PnP 结果（包含被质量门限拒绝的结果），供 L6 调试。
  const std::vector<Armor>& observations() const noexcept;

  // 当前 EKF 整车模型生成的物理装甲板 [x, y, z, yaw]。这是只读调试
  // 快照，供离线回放和 L6 可视化使用；Lost 状态返回空数组。
  std::vector<Eigen::Vector4d> targetArmorPoses() const;

  // 清空状态机、时间连续性记录和本帧观测。
  void reset() noexcept;

private:
  // 筛掉 PnP 未成功提交的观测，并按图像中心距离生成候选列表。
  bool observationUsable(const Armor& armor) const noexcept;
  std::vector<const Armor*> usableObservations() const;
  // 分别处理 Lost 状态初始化和已有目标的预测、观测更新。
  bool initializeTarget(
    const std::vector<const Armor*>& observations,
    TimePoint timestamp);
  bool updateTarget(
    const std::vector<const Armor*>& observations,
    TimePoint timestamp);
  // 根据本帧是否成功关联观测推进四态状态机。
  void updateState(bool found);
  // 只重置跟踪生命周期；帧时间和诊断观测由调用方按需清理。
  void resetTracking() noexcept;

  ArmorConfig armor_config_;
  TrackerConfig tracker_config_;
  TargetConfig target_config_;
  PnpSolver pnp_solver_;
  cv::Point2f image_center_{};
  bool ready_{false};

  // 状态机计数、当前目标、上一帧时间及本帧全部观测。
  TrackState state_{TrackState::Lost};
  int detect_count_{0};
  std::uint64_t reset_count_{0};
  int temp_lost_count_{0};
  std::optional<TrackedTarget> target_;
  std::optional<TimePoint> last_timestamp_;
  std::vector<Armor> observations_;
};

}  // namespace L3Estimation
