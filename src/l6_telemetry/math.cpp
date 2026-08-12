#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

std::chrono::steady_clock::duration toDuration(double seconds)
{
  return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::microseconds(static_cast<std::int64_t>(seconds * 1e6)));
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
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  return {
    std::atan2(y, x), std::atan2(z, std::sqrt(x * x + y * y)),
    std::sqrt(x * x + y * y + z * z)};
}

Eigen::Matrix3d xyz2ypdJacobian(const Eigen::Vector3d& xyz)
{
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  const double xy_squared = x * x + y * y;
  const double distance_squared = xy_squared + z * z;

  if (xy_squared <= 1e-12 || distance_squared <= 1e-12) {
    return Eigen::Matrix3d::Zero();
  }

  const double xy = std::sqrt(xy_squared);
  const double distance = std::sqrt(distance_squared);

  Eigen::Matrix3d jacobian;
  jacobian <<
    -y / xy_squared, x / xy_squared, 0.0,
    -x * z / (distance_squared * xy), -y * z / (distance_squared * xy), xy / distance_squared,
    x / distance, y / distance, z / distance;
  return jacobian;
}

//gtsam

inline double logisticFunction(double x) {
  if (x > 0)
    return 1.0 / (1.0 + std::exp(-x));
  else
    return std::exp(x) / (1.0 + std::exp(x));
}

inline double logisticFunction(double x, double min, double max) {
  return (logisticFunction(x) * (max - min)) + min;
}

inline double logisticInverse(double y, double min, double max) {
  return std::log((y - min) / (max - y));
}

// NOTE: 需要传入的是逻辑函数的输出值
inline double logisticDerivative(double y) { return y * (1.0 - y); }

inline double logisticDerivative(double y, double min, double max) {
  return (y - min) * (max - y) / (max - min);
}

inline double signalFunction(double x) {
  if (x > 1e-6)
    return 1;
  if (x < 1e-6)
    return -1;
  return 0;
}

inline double errorFunction(double x, double min, double max) {
  return (std::erf(x) + 1.0) / 2.0 * (max - min) + min;
}

inline double errorInverse(double y) {
  constexpr double a = 8 * (std::numbers::pi - 3) /
                       (3 * std::numbers::pi * (4 - std::numbers::pi));
  constexpr double inv_a = 1 / a;
  double b = 2 * std::numbers::inv_pi * inv_a;
  double c = std::log(1 - y * y);
  // NOTE: 为误差函数反函数的解析近似，详见
  // https://zh.wikipedia.org/wiki/%E8%AF%AF%E5%B7%AE%E5%87%BD%E6%95%B0
  double x =
      signalFunction(y) *
      std::sqrt(std::sqrt((b + c / 2) * (b + c / 2) - c / a) - (b + c / 2));
  return x;
}

inline double errorInverse(double y, double min, double max) {
  return errorInverse((2.0 * y - 2 * min) / (max - min) - 1);
}

inline double standardNormalCDF(double x, double standard_deviation = 1) {
  return errorFunction(x / (standard_deviation * std::numbers::sqrt2), 0, 1);
}




}  // namespace L6Telemetry
