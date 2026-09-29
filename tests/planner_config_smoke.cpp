#include "l4_planning/planner_config.hpp"
#include "l4_planning/window_policy.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>

namespace {

[[nodiscard]] double toDegrees(double radians) noexcept
{
  return radians * 180.0 / std::numbers::pi;
}

}  // namespace

int main()
{
  const L4Planning::PlannerTuning tuning =
    L4Planning::loadPlannerTuning("config/planner_config.yaml");
  const L4Planning::PlannerConfig& planner = tuning.planner;

  bool fields_loaded = true;
  const auto expect = [&fields_loaded](bool condition, const char* name) {
    if (!condition) {
      std::cerr << "Unexpected planner field: " << name << '\n';
      fields_loaded = false;
    }
  };
  const auto near = [](double actual, double expected) {
    return std::abs(actual - expected) <= 1e-12;
  };
  expect(planner.max_iterations == 20, "max_iterations");
  expect(
    planner.fly_time_tolerance == std::chrono::microseconds{200},
    "fly_time_tolerance");
  expect(near(planner.position_tolerance, 0.005), "position_tolerance");
  expect(near(tuning.default_bullet_speed, 27.0), "default_bullet_speed");
  expect(
    near(tuning.latency.high_speed_fire_delay, 0.030),
    "high_speed_fire_delay");
  expect(
    near(tuning.latency.low_speed_fire_delay, 0.015),
    "low_speed_fire_delay");
  expect(near(tuning.latency.decision_speed, 8.0), "decision_speed");
  expect(near(planner.switch_yaw_dead_zone, 5.0), "switch_yaw_dead_zone");
  expect(
    near(planner.switch_pitch_dead_zone, 2.5),
    "switch_pitch_dead_zone");
  expect(planner.enable_dynamic_windows, "enable_dynamic_windows");
  expect(
    near(planner.window_shrink_start_speed, 2.0),
    "window_shrink_start_speed");
  expect(
    near(planner.window_shrink_end_speed, 6.0),
    "window_shrink_end_speed");
  expect(
    near(planner.selection_min_window_scale, 0.75),
    "selection_min_window_scale");
  expect(
    near(planner.firing_max_window_scale, 0.90),
    "firing_max_window_scale");
  expect(
    near(planner.firing_min_window_scale, 0.50),
    "firing_min_window_scale");
  expect(
    near(planner.selection_hold_margin, 8.0),
    "selection_hold_margin");
  expect(!planner.enable_mpc, "enable_mpc");
  expect(planner.mpc_max_iterations == 10, "mpc_max_iterations");
  expect(near(planner.mpc_admm_rho, 1.0), "mpc_admm_rho");
  expect(
    near(planner.min_yaw_acceleration, -50.0),
    "min_yaw_acceleration");
  expect(
    near(planner.max_pitch_acceleration, 100.0),
    "max_pitch_acceleration");
  expect(
    near(tuning.armor_score_weights.facing_weight, 0.40),
    "score_weights.facing");
  expect(
    near(tuning.armor_score_weights.window_weight, 0.35),
    "score_weights.window");
  expect(
    near(tuning.armor_score_weights.aim_cost_weight, 0.25),
    "score_weights.aim_cost");
  if (!fields_loaded) {
    return 1;
  }

  const L4Planning::DynamicWindows low =
    L4Planning::computeDynamicWindows(75.0, 45.0, 1.0, planner);
  const L4Planning::DynamicWindows middle =
    L4Planning::computeDynamicWindows(75.0, 45.0, 5.0, planner);
  const L4Planning::DynamicWindows high =
    L4Planning::computeDynamicWindows(75.0, 45.0, -10.0, planner);
  if (std::abs(toDegrees(low.selection.enter) - 75.0) > 1e-12
      || std::abs(toDegrees(low.firing.enter) - 67.5) > 1e-12
      || std::abs(toDegrees(low.hold.enter) - 83.0) > 1e-12
      || std::abs(toDegrees(middle.selection.enter) - 60.9375) > 1e-12
      || std::abs(toDegrees(middle.firing.leave) - 27.0) > 1e-12
      || std::abs(toDegrees(high.selection.enter) - 56.25) > 1e-12
      || std::abs(toDegrees(high.firing.enter) - 37.5) > 1e-12
      || !(low.hold.enter > low.selection.enter
           && low.selection.enter > low.firing.enter
           && middle.hold.leave > middle.selection.leave
           && middle.selection.leave > middle.firing.leave
           && high.hold.enter > high.selection.enter
           && high.selection.enter > high.firing.enter)) {
    std::cerr << "Dynamic windows are not linear, clamped, and nested\n";
    return 2;
  }

  std::cout << "Planner config smoke test passed\n";
  return 0;
}
