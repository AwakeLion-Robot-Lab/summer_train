#pragma once

#include "l3_estimation/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace L3Estimation {

// 两种估计后端共同遵守的物理状态契约。这里只描述“估什么”，不包含 EKF、
// GTSAM Key、因子或优化器等任何后端实现细节。
inline constexpr int kTargetStateSize = 11;

enum TargetStateIndex : int {
  CenterX = 0,
  VelocityX = 1,
  CenterY = 2,
  VelocityY = 3,
  CenterZ = 4,
  VelocityZ = 5,
  Yaw = 6,
  Vyaw = 7,
  RadiusA = 8,
  RadiusDifference = 9,
  HeightDifference = 10
};

using TargetStateVector = Eigen::Matrix<double, kTargetStateSize, 1>;
using TargetCovariance = Eigen::Matrix<double, kTargetStateSize, kTargetStateSize>;

// 后端中立的恒速度/恒角速度模型。EKF 用它传播滤波状态，L4 也用同一个模型
// 对只读估计快照做延迟补偿。
[[nodiscard]] TargetCovariance targetTransition(double dt) noexcept;
[[nodiscard]] TargetCovariance targetProcessNoise(
  double dt,
  bool outpost,
  const TargetConfig& config) noexcept;
[[nodiscard]] TargetStateVector predictTargetState(
  const TargetStateVector& state,
  double dt) noexcept;

[[nodiscard]] Eigen::Vector3d armorPosition(
  const TargetStateVector& state,
  int armor_count,
  int armor_id) noexcept;
[[nodiscard]] std::vector<Eigen::Vector4d> armorPoses(
  const TargetStateVector& state,
  int armor_count);

}  // namespace L3Estimation
