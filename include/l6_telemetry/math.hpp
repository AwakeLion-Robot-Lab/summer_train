#ifndef TOOLS__MATH_HPP
#define TOOLS__MATH_HPP

#include <Eigen/Geometry>

#include <chrono>

namespace L6Telemetry {

// 计算时间差a - b，单位：s
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw);

Eigen::Quaterniond slerpQuaternion(
  const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double k);

}  // namespace L6Telemetry

#endif  // TOOLS__MATH_HPP
