#pragma once

#include "rune/f4-planning/planning.hpp"
#include "rune/f4-planning/ballistic_solver.hpp"

namespace Rune::detail {

struct Intercept {
  BallisticSolution ballistic;
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};
  double horizon_s{0};
};

[[nodiscard]] std::optional<Intercept> intercept(
    const Motion& motion, const Eigen::Vector3d& point_aim,
    const Eigen::Vector3d& center_aim, const Eigen::Vector3d& normal_aim,
    bool center_hold, const ControlInput& control, const PlanningConfig& config,
    double command_offset_s = 0.0);

// Updates feedforward and optionally smooths a centered reference via TinyMPC.
void optimizeTrajectory(AimPlan& plan, const Motion& motion,
    const Eigen::Vector3d& initial_point, const Eigen::Vector3d& normal_aim,
    const ControlInput& control, const PlanningConfig& config);

} // namespace Rune::detail
