#include "l4_planning/ballistic_solver.hpp"

#include <cmath>

namespace L4Planning {
namespace {

// 等效射程 A = (e^{k·d} - 1)/k。k -> 0 时是 0/0，用真空解的极限收尾。
[[nodiscard]] double effectiveRange(double distance, double drag) noexcept
{
  return drag < 1e-6 ? distance : std::expm1(drag * distance) / drag;
}

}  // namespace

std::optional<Ballistic> solveBallistic(
  double distance, double height, double bullet_speed, const BallisticConfig& config)
{
  if (!std::isfinite(distance) || !std::isfinite(height) || !std::isfinite(bullet_speed) ||
      distance <= 0.0 || bullet_speed < 1e-6) {
    return std::nullopt;
  }

  const double a = effectiveRange(distance, config.drag_coefficient);
  const double b = config.gravity * a * a / (2.0 * bullet_speed * bullet_speed);
  const double discriminant = a * a - 4.0 * b * (b + height);
  if (b < 1e-12 || discriminant < 0.0) {
    return std::nullopt;
  }

  const double sqrt_discriminant = std::sqrt(discriminant);
  const double pitch_low = std::atan((a - sqrt_discriminant) / (2.0 * b));
  const double pitch_high = std::atan((a + sqrt_discriminant) / (2.0 * b));

  const auto flight_time = [a, bullet_speed](double pitch) {
    return a / (bullet_speed * std::cos(pitch));
  };
  const double time_low = flight_time(pitch_low);
  const double time_high = flight_time(pitch_high);
  const bool take_low = time_low <= time_high;

  Ballistic result;
  result.pitch = take_low ? pitch_low : pitch_high;
  result.fly_time = take_low ? time_low : time_high;
  if (!std::isfinite(result.pitch) || !std::isfinite(result.fly_time) ||
      result.fly_time <= 0.0 || std::abs(result.pitch) > config.max_pitch) {
    return std::nullopt;
  }
  return result;
}

}  // namespace L4Planning
