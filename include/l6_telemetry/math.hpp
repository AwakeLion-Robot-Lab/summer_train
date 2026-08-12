#pragma once

#include <Eigen/Geometry>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <array>
#include <chrono>

namespace L6Telemetry {

// 计算时间差 a - b，单位：s
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

// 秒 -> steady_clock 时长，取微秒截断：延迟和飞行时间都在毫秒量级，
// 亚微秒的尾数没有物理意义，截断掉可以让回放逐帧可复现。
[[nodiscard]] std::chrono::steady_clock::duration toDuration(double seconds);

// 把角度归一化到 (-pi, pi]。
double limit_rad(double angle);

[[nodiscard]] Eigen::Matrix3d toEigen(
  const cv::Matx33d& rotation) noexcept;

[[nodiscard]] cv::Matx33d toCv(
  const Eigen::Matrix3d& rotation) noexcept;

// 使用鞋带公式计算四边形的像素面积。
[[nodiscard]] double polygonArea(
  const std::array<cv::Point2f, 4>& corners) noexcept;

[[nodiscard]] double squaredDistance(
  const cv::Point2f& lhs,
  const cv::Point2f& rhs) noexcept;

// 欧拉角固定使用 [yaw, pitch, roll]；axis0/1/2 指定旋转轴顺序。
Eigen::Vector3d eulers(
  Eigen::Quaterniond quaternion,
  int axis0,
  int axis1,
  int axis2,
  bool extrinsic = false);

Eigen::Vector3d eulers(
  const Eigen::Matrix3d& rotation,
  int axis0,
  int axis1,
  int axis2,
  bool extrinsic = false);

// YPR 固定为 [yaw, pitch, roll]，旋转顺序为 Rz(yaw)Ry(pitch)Rx(roll)。
Eigen::Matrix3d yprToRotation(const Eigen::Vector3d& ypr);

Eigen::Vector3d rotationToYpr(const Eigen::Matrix3d& rotation);

// RPY 固定为 [roll, pitch, yaw]，旋转顺序为 Rz(yaw)Ry(pitch)Rx(roll)。
Eigen::Matrix3d rpyToRotation(const Eigen::Vector3d& rpy);

Eigen::Vector3d rotationToRpy(const Eigen::Matrix3d& rotation);

// 笛卡尔 [x, y, z] -> 球坐标 [方位角, 俯仰角, 距离]，以及它的 Jacobian。
// Jacobian 在原点和 z 轴上不可导，退化时返回零矩阵，让上层的观测不产生修正。
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);

Eigen::Matrix3d xyz2ypdJacobian(const Eigen::Vector3d& xyz);

Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw);

Eigen::Quaterniond slerpQuaternion(
  const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double k);

}  // namespace L6Telemetry
