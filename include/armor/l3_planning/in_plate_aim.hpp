#pragma once

#include "l3_planning/predictor.hpp"
#include "l3_planning/types.hpp"

namespace L4Planning {

// 查询不修改快照，候选、弹道迭代和 MPC 可任意重复调用。
[[nodiscard]] InPlateAimPoint queryInPlateAim(
  const PredictionResult& prediction,
  const ArmorPose& armor,
  const InPlateAimState& state,
  TimePoint control_time,
  double gun_yaw,
  const InPlateAimConfig& config,
  const Eigen::Vector3d& gun_origin_world = Eigen::Vector3d::Zero());

// yaw 沿板号递增；正向旋转时较小的相邻板号先接替，停转返回 -1。
[[nodiscard]] int nextArmorInRotation(int armor_id, int armor_count, int direction) noexcept;

[[nodiscard]] bool validInPlateAimConfig(const InPlateAimConfig& config) noexcept;

}  // namespace L4Planning
