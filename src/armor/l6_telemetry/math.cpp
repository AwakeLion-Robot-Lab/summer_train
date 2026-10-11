#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace L6Telemetry {

Eigen::Matrix3d toEigen(const cv::Matx33d& rotation) noexcept
{
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = rotation(row, col);
    }
  }
  return result;
}

cv::Matx33d toCv(const Eigen::Matrix3d& rotation) noexcept
{
  cv::Matx33d result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = rotation(row, col);
    }
  }
  return result;
}

double polygonArea(const std::array<cv::Point2f, 4>& corners) noexcept
{
  double twice_signed_area = 0.0;
  for (std::size_t index = 0; index < corners.size(); ++index) {
    const auto& current = corners[index];
    const auto& next = corners[(index + 1) % corners.size()];
    twice_signed_area +=
      static_cast<double>(current.x) * static_cast<double>(next.y) -
      static_cast<double>(current.y) * static_cast<double>(next.x);
  }
  return std::abs(twice_signed_area) * 0.5;
}

double squaredDistance(
  const cv::Point2f& lhs,
  const cv::Point2f& rhs) noexcept
{
  const double dx = static_cast<double>(lhs.x) - static_cast<double>(rhs.x);
  const double dy = static_cast<double>(lhs.y) - static_cast<double>(rhs.y);
  return dx * dx + dy * dy;
}

Eigen::Vector3d eulers(
  Eigen::Quaterniond quaternion,
  int axis0,
  int axis1,
  int axis2,
  bool extrinsic)
{
  if (!extrinsic) {
    std::swap(axis0, axis2);
  }

  const int i = axis0;
  const int j = axis1;
  int k = axis2;
  const bool proper_euler = i == k;
  if (proper_euler) {
    k = 3 - i - j;
  }
  const int sign = (i - j) * (j - k) * (k - i) / 2;

  const Eigen::Vector4d xyzw = quaternion.coeffs();
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  double d = 0.0;
  if (proper_euler) {
    a = xyzw[3];
    b = xyzw[i];
    c = xyzw[j];
    d = xyzw[k] * sign;
  } else {
    a = xyzw[3] - xyzw[j];
    b = xyzw[i] + xyzw[k] * sign;
    c = xyzw[j] + xyzw[3];
    d = xyzw[k] * sign - xyzw[i];
  }

  const double norm_squared = a * a + b * b + c * c + d * d;
  if (!std::isfinite(norm_squared) || norm_squared <= 1e-12) {
    return Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  }

  Eigen::Vector3d angles;
  const double acos_argument = std::clamp(
    2.0 * (a * a + b * b) / norm_squared - 1.0, -1.0, 1.0);
  angles[1] = std::acos(acos_argument);

  const double half_sum = std::atan2(b, a);
  const double half_difference = std::atan2(-d, c);
  constexpr double kSingularityTolerance = 1e-7;
  const bool away_from_zero =
    std::abs(angles[1]) >= kSingularityTolerance;
  const bool away_from_pi =
    std::abs(angles[1] - std::numbers::pi) >= kSingularityTolerance;

  if (away_from_zero && away_from_pi) {
    angles[0] = half_sum + half_difference;
    angles[2] = half_sum - half_difference;
  } else if (!extrinsic) {
    angles[0] = 0.0;
    angles[2] = away_from_zero ? -2.0 * half_difference
                               : 2.0 * half_sum;
  } else {
    angles[2] = 0.0;
    angles[0] = away_from_zero ? 2.0 * half_difference
                               : 2.0 * half_sum;
  }

  for (double& angle : angles) {
    angle = limit_rad(angle);
  }

  if (!proper_euler) {
    angles[2] *= sign;
    angles[1] -= std::numbers::pi / 2.0;
  }
  if (!extrinsic) {
    std::swap(angles[0], angles[2]);
  }
  return angles;
}

Eigen::Vector3d eulers(
  const Eigen::Matrix3d& rotation,
  int axis0,
  int axis1,
  int axis2,
  bool extrinsic)
{
  return eulers(
    Eigen::Quaterniond(rotation), axis0, axis1, axis2, extrinsic);
}

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

Eigen::Matrix3d yprToRotation(const Eigen::Vector3d& ypr)
{
  return (
    Eigen::AngleAxisd(ypr.x(), Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(ypr.y(), Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(ypr.z(), Eigen::Vector3d::UnitX()))
    .toRotationMatrix();
}

Eigen::Vector3d rotationToYpr(const Eigen::Matrix3d& rotation)
{
  return eulers(rotation, 2, 1, 0);
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
