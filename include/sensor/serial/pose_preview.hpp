#pragma once
#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <optional>

namespace L1Sensor {
struct TimedGimbalPose {
  Eigen::Quaterniond rotation;
  std::chrono::steady_clock::time_point timestamp;
};

// Only a provisional ROI pose, never a replacement for the exposure pose.
// Reject batched/irregular packets, stale samples and fast rotation. If history
// already covers the image, the normal interpolated path needs no waiting.
inline std::optional<Eigen::Quaterniond> previewRoiPose(
    const TimedGimbalPose& before, const TimedGimbalPose& latest,
    std::chrono::steady_clock::time_point image_time) {
  const double age = std::chrono::duration<double>(image_time - latest.timestamp).count();
  const double dt = std::chrono::duration<double>(latest.timestamp - before.timestamp).count();
  if (age <= 0 || age > 0.008 || dt < 0.0005 || dt > 0.020 ||
      !before.rotation.coeffs().allFinite() || !latest.rotation.coeffs().allFinite() ||
      before.rotation.squaredNorm() < 1e-12 || latest.rotation.squaredNorm() < 1e-12)
    return std::nullopt;
  const auto q = latest.rotation.normalized();
  if (before.rotation.normalized().angularDistance(q) / dt > 1.0) return std::nullopt;
  return q;
}
} // namespace L1Sensor
