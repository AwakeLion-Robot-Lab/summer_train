#ifndef TOOLS__MATH_HPP
#define TOOLS__MATH_HPP

#include <Eigen/Geometry>

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

#include <array>
#include <chrono>

namespace L6Telemetry {

// 计算时间差a - b，单位：s
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

double limit_rad(double angle);

Eigen::Matrix3d toEigen(
  const cv::Matx33d& rotation) noexcept;

cv::Matx33d toCv(
  const Eigen::Matrix3d& rotation) noexcept;

// 使用鞋带公式计算四边形的像素面积。
double polygonArea(
  const std::array<cv::Point2f, 4>& corners) noexcept;

double squaredDistance(
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

Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);

Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw);

Eigen::Quaterniond slerpQuaternion(
  const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double k);

}  // namespace L6Telemetry

#endif  // TOOLS__MATH_HPP
