#pragma once

#include "l2_perception/armor.hpp"
#include "l3_estimation/gtsam_est/config.hpp"
#include "l3_estimation/tracked_target.hpp"
#include "l3_estimation/tracker.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Geometry>

#include <memory>
#include <optional>
#include <vector>

// 因子图后端：GTSAM + ISAM2 增量优化。移植自 jlu_vision_26-master 的
// src/auto_aim/armor_tracker（factors.hpp / target.hpp）。
//
// 与 filter_est 的本质差别：EKF 每帧把历史压缩进一个高斯，装甲板关联错了就再也
// 改不回来；因子图保留当前跟踪段的原始约束，ISAM2 在重线性化时能把早先的错误
// 关联一起修正。与 JLU 一致，只在目标 LOST 时清空整张图。
//
// **本头文件不包含任何 gtsam 头**：所有 gtsam 类型都藏在 Impl 里（见
// src/l3_estimation/gtsam_est/tracker.cpp），这样没装 GTSAM 的机器照样能编译
// 整个工程和 makeTracker() 工厂。
namespace L3Estimation::GtsamEst {

// 本次构建是否把 GTSAM 实现编进来了（等价于 xmake f --use_gtsam=y）。
[[nodiscard]] bool available() noexcept;

class Tracker : public ITracker {
public:
  // GTSAM 未编入时**直接抛 std::runtime_error**，不静默回退到 EKF：回退会让
  // 回放曲线看起来是因子图跑出来的，实际是卡尔曼的结果，这种误导比起不来更贵。
  // 标定无效则不是错误，照常构造但 ready() 为 false。
  explicit Tracker(
    const L1Sensor::CameraCalibration& calibration,
    ArmorConfig armor_config = {},
    TargetConfig target_config = {},
    Config gtsam_config = {});
  ~Tracker() override;

  [[nodiscard]] bool ready() const noexcept override;
  [[nodiscard]] TrackState state() const noexcept override;
  [[nodiscard]] EstimatorBackend backend() const noexcept override;

  [[nodiscard]] std::optional<TrackedTarget> track(
    const std::vector<L2Perception::Armor>& detections,
    const std::optional<Eigen::Quaterniond>& q_world_barrel,
    TimePoint timestamp) override;

  [[nodiscard]] const std::vector<Armor>& observations() const noexcept override;
  [[nodiscard]] std::vector<Eigen::Vector4d> targetArmorPoses() const override;
  void reset() noexcept override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace L3Estimation::GtsamEst
