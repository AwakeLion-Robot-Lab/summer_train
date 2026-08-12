#pragma once

#include "l3_estimation/target_estimator.hpp"

#include <Eigen/Core>

#include <vector>

namespace L4Planning {

// 整车模型外推。做法与 sp_vision 的 Aimer 一致：在目标的**副本**上调用 L3
// 自己的 predict()，外推用的就是 EKF 的状态转移模型，不会出现两份需要手工
// 同步的实现。dt 允许为负，用于把状态回退到过去时刻做对照。
//
// 关键点：装甲板的位置由 (旋转中心, 整车 yaw, 半径) 共同决定，因此外推
// **必须同时推进中心和 yaw**。只推中心不推 yaw 时，小陀螺目标会被算成
// 原地不动——v_yaw 取 10 rad/s 时，100 ms 的延迟就对应 57 度的偏差，
// 这正是延迟补偿要解决的主要误差来源。
class Predictor {
public:
  // 恒速度 + 恒角速度外推，同时推进滤波器时刻，与 sp 的 target.predict(future)
  // 等价。协方差也会按 Q 传播，因为复用的就是 L3 的 predict。
  [[nodiscard]] L3Estimation::TrackedTarget predict(
    const L3Estimation::TrackedTarget& target, double dt) const;

  // 先外推 dt，再把整车模型展开成全部物理装甲板的 [x, y, z, yaw]。
  // 顺序与 L3 的 armor_xyza_list() 一致，下标即物理装甲板编号。
  [[nodiscard]] std::vector<Eigen::Vector4d> armorPosesAt(
    const L3Estimation::TrackedTarget& target, double dt) const;

  // 展开给定状态的装甲板，不做外推。armorPosesAt 的内部实现。
  [[nodiscard]] std::vector<Eigen::Vector4d> armorPoses(
    const L3Estimation::TrackedTarget& target) const;
};

}  // namespace L4Planning
