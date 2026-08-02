#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace L6Telemetry {

double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b)
{
  std::chrono::duration<double> c = a - b;
  return c.count();
}

double limit_rad(double angle)
{
  if (!std::isfinite(angle)) {
    return angle;
  }

  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  double limited = std::remainder(angle, kTwoPi);
  if (limited <= -std::numbers::pi) {
    limited += kTwoPi;
  }
  return limited;
}

Eigen::Matrix3d rpyToRotation(const Eigen::Vector3d& rpy)
{
  return (
    Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()))
    .toRotationMatrix();
}

Eigen::Vector3d rotationToRpy(const Eigen::Matrix3d& rotation)
{
  // Eigen 返回 [yaw, pitch, roll]，对外统一为 [roll, pitch, yaw]。
  const Eigen::Vector3d ypr = rotation.eulerAngles(2, 1, 0);
  return {ypr.z(), ypr.y(), ypr.x()};
}

Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw)
{
  return Eigen::Quaterniond(rpyToRotation({roll, pitch, yaw})).normalized();
}

Eigen::Quaterniond slerpQuaternion(
  const Eigen::Quaterniond& a, const Eigen::Quaterniond& b, double k)
{
  const double ratio = std::clamp(k, 0.0, 1.0);
  return a.normalized().slerp(ratio, b.normalized()).normalized();
}

Eigen::Vector3d xyz2ypd(const Eigen::Vector3d& xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];
  auto yaw = std::atan2(y, x);
  auto pitch = std::atan2(z, std::sqrt(x * x + y * y));
  auto distance = std::sqrt(x * x + y * y + z * z);
  return {yaw, pitch, distance};
}

}  // namespace L6Telemetry
