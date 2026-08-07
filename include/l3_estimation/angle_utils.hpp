#pragma once

#include <Eigen/Core>

#include <cmath>
#include <numbers>

namespace L3Estimation {

// 把任意角度归一化到 (-pi, pi]。
[[nodiscard]] inline double normalizeAngle(double angle) noexcept
{
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  angle = std::remainder(angle, kTwoPi);
  return angle <= -std::numbers::pi ? angle + kTwoPi : angle;
}

// 检查是否为合法旋转矩阵：有限、正交（RᵀR ≈ I）、行列式为 1。
[[nodiscard]] inline bool isRotationMatrix(
  const Eigen::Matrix3d& rotation) noexcept
{
  if (!rotation.allFinite()) {
    return false;
  }
  const Eigen::Matrix3d error =
    rotation.transpose() * rotation - Eigen::Matrix3d::Identity();
  return error.norm() < 1e-5
         && std::abs(rotation.determinant() - 1.0) < 1e-5;
}

}  // namespace L3Estimation
