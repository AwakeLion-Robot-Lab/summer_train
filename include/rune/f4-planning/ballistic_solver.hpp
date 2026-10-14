#pragma once

#include <Eigen/Core>

#include <optional>

namespace Rune::detail {

struct BallisticSolution {
  double yaw;
  double pitch;
  double flight_time;
};

[[nodiscard]] std::optional<BallisticSolution> solveBallistic(
  const Eigen::Vector3d& target_barrel, double speed, double gravity);

}  // namespace Rune::detail
