#include "rune/f4-planning/ballistic_solver.hpp"

#include <cmath>

namespace Rune::detail {

std::optional<BallisticSolution> solveBallistic(
  const Eigen::Vector3d& target, double speed, double gravity) {
  if (!target.allFinite() || !std::isfinite(speed) || !std::isfinite(gravity) ||
      speed <= 0 || gravity <= 0 || target.x() <= 0) return std::nullopt;
  const double horizontal = std::hypot(target.x(), target.y());
  const double speed2 = speed * speed;
  const double discriminant = speed2 * speed2 - gravity *
    (gravity * horizontal * horizontal + 2 * target.z() * speed2);
  if (horizontal <= 0 || discriminant < 0 || !std::isfinite(discriminant))
    return std::nullopt;
  const double pitch = std::atan((speed2 - std::sqrt(discriminant)) /
                                 (gravity * horizontal));
  const double flight_time = horizontal / (speed * std::cos(pitch));
  if (!std::isfinite(flight_time) || flight_time <= 0) return std::nullopt;
  return BallisticSolution{std::atan2(target.y(), target.x()), pitch,
                           flight_time};
}

}  // namespace Rune::detail
