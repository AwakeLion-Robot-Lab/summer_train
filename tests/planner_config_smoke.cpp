#include "l4_planning/planner_config.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

int main()
{
  const L4Planning::PlannerTuning tuning =
    L4Planning::loadPlannerTuning("config/planner_config.yaml");
  const L4Planning::PlannerConfig& planner = tuning.planner;

  if (planner.max_iterations != 20
      || planner.fly_time_tolerance != std::chrono::microseconds{200}
      || std::abs(planner.position_tolerance - 0.005) > 1e-12
      || !planner.enable_mpc
      || planner.mpc_max_iterations != 10
      || planner.mpc_admm_rho != 1.0
      || planner.min_yaw_acceleration != -50.0
      || planner.max_pitch_acceleration != 100.0
      || tuning.armor_score_weights.facing_weight != 0.40
      || tuning.armor_score_weights.window_weight != 0.50
      || tuning.armor_score_weights.aim_cost_weight != 0.10) {
    std::cerr << "Planner YAML fields were not loaded correctly\n";
    return 1;
  }

  std::cout << "Planner config smoke test passed\n";
  return 0;
}
