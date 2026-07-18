#ifndef TOOLS__MATH_HPP
#define TOOLS__MATH_HPP

#include <Eigen/Geometry>

#include <chrono>

namespace L6Telemetry {

// 计算时间差a - b，单位：s
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

double limit_rad(double angle);

// RPY 固定为 [roll, pitch, yaw]，旋转顺序为 Rz(yaw)Ry(pitch)Rx(roll)。
Eigen::Matrix3d rpyToRotation(const Eigen::Vector3d& rpy);

Eigen::Vector3d rotationToRpy(const Eigen::Matrix3d& rotation);

Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz);

Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw);

Eigen::Quaterniond slerpQuaternion(
  const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double k);

}  // namespace L6Telemetry

#endif  // TOOLS__MATH_HPP
