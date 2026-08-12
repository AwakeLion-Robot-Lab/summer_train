#pragma once

#include "l3_estimation/tracked_target.hpp"

#include <Eigen/Core>

#include <vector>

namespace L4Planning {

// 整车模型外推。做法是在后端中立快照的副本上调用公共运动学模型，不会把 L4
// 绑到 EKF 或 GTSAM 的内部对象。dt 允许为负，用于把状态回退到过去时刻做对照。
//
// 关键点：装甲板位置由 (旋转中心, 整车 yaw, 半径) 共同决定，所以外推**必须同时
// 推进中心和 yaw**。只推中心不推 yaw 时，小陀螺目标会被算成原地不动——v_yaw
// 取 10 rad/s 时，100 ms 延迟就对应 57 度偏差，这正是延迟补偿的主要误差来源。
class Predictor {
public:
  [[nodiscard]] L3Estimation::TrackedTarget predict(
    const L3Estimation::TrackedTarget& target, double dt) const;

  // 先外推 dt，再把整车模型展开成全部物理装甲板的 [x, y, z, yaw]。
  [[nodiscard]] std::vector<Eigen::Vector4d> armorPosesAt(
    const L3Estimation::TrackedTarget& target, double dt) const;

  // 展开给定状态的装甲板，不做外推。
  [[nodiscard]] std::vector<Eigen::Vector4d> armorPoses(
    const L3Estimation::TrackedTarget& target) const;
};

}  // namespace L4Planning
