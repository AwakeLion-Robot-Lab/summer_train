#include "l4_planning/buff/trajectory_solver.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace L4Planning::Buff {
namespace {

constexpr double kGravity = 9.81;
constexpr double kDrag = 0.003;
constexpr double kDt = 0.005;
constexpr double kTimeout = 4.0;
constexpr double kMinHorizontalSpeed = 0.1;
constexpr double kMaxPitch = 80.0 / 57.3;
constexpr double kHeightError = 0.001;
constexpr int kMaxIterations = 10;

}  // namespace

std::optional<TrajectoryOutput> TrajectorySolver::solve(
  const Eigen::Vector3d& point, double bullet_speed) const
{
  const double distance = std::hypot(point.x(), point.y());
  if (!point.allFinite() || !std::isfinite(bullet_speed) ||
      distance <= 0.0 || bullet_speed <= 0.0) return std::nullopt;

  const double yaw = std::atan2(point.y(), point.x());
  double pitch = std::atan2(point.z(), distance);

  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    double x = 0.0;
    double z = 0.0;
    double vx = bullet_speed * std::cos(pitch);
    double vz = bullet_speed * std::sin(pitch);
    double time = 0.0;
    double previous_x = 0.0;
    double previous_z = 0.0;
    double previous_time = 0.0;
    while (x < distance) {
      previous_x = x;
      previous_z = z;
      previous_time = time;
      const double speed = std::hypot(vx, vz);
      vx -= kDrag * speed * vx * kDt;
      vz -= (kGravity + kDrag * speed * vz) * kDt;
      x += vx * kDt;
      z += vz * kDt;
      time += kDt;
      if (time > kTimeout || vx <= kMinHorizontalSpeed) break;
    }
    if (x >= distance && x > previous_x) {
      const double ratio = (distance - previous_x) / (x - previous_x);
      z = std::lerp(previous_z, z, ratio);
      time = std::lerp(previous_time, time, ratio);
    }
    const double error = point.z() - z;
    if (std::abs(error) < kHeightError) {
      return TrajectoryOutput{time, yaw, -pitch};
    }
    pitch += std::atan2(error, distance);
    if (!std::isfinite(pitch) || std::abs(pitch) > kMaxPitch) return std::nullopt;
  }
  return std::nullopt;
}

}  // namespace L4Planning::Buff
