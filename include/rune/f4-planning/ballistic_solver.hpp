#pragma once

#include <Eigen/Core>

#include <optional>

namespace Rune::detail {

struct BallisticSolution {
  double yaw;
  double pitch;
  double flight_time;
  bool used_air_resistance{false};
};

[[nodiscard]] std::optional<BallisticSolution> solveBallistic(
  const Eigen::Vector3d& target_barrel, double speed, double gravity,
  bool enable_air_resistance = false, double linear_drag_coefficient = 0.0);

}  // namespace Rune::detail
