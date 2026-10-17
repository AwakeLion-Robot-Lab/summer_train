#include "rune/f4-planning/ballistic_solver.hpp"
#include "armor/l3_planning/ballistic_solver.hpp"

namespace Rune::detail {

std::optional<BallisticSolution> solveBallistic(
    const Eigen::Vector3d& target, double speed, double gravity,
    bool enable_air_resistance, double linear_drag_coefficient) {
  if (!target.allFinite()) return {};
  L4Planning::BallisticRequest request;
  request.target_position_barrel=target;
  request.bullet_speed=speed;
  request.gravity=gravity;
  request.enable_air_resistance=enable_air_resistance;
  request.linear_drag_coefficient=linear_drag_coefficient;
  const auto solution=L4Planning::BallisticSolver{}.solve(request);
  if (!solution.valid) return {};
  return BallisticSolution{solution.yaw,solution.pitch,solution.fly_time,
                           solution.used_air_resistance};
}

} // namespace Rune::detail
