#include "l4_planning/window_policy.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace L4Planning {

DynamicWindows computeDynamicWindows(
  double base_enter_angle_degree,
  double base_leave_angle_degree,
  double relative_yaw_rate,
  const PlannerConfig& config) noexcept
{
  constexpr double kDegreeToRad = std::numbers::pi / 180.0;
  const double base_enter = base_enter_angle_degree * kDegreeToRad;
  const double base_leave = base_leave_angle_degree * kDegreeToRad;

  if (!config.enable_dynamic_windows) {
    const AngularWindow base{base_enter, base_leave};
    return {base, base, base};
  }

  const double speed = std::abs(relative_yaw_rate);
  const double progress = std::clamp(
    (speed - config.window_shrink_start_speed)
      / (config.window_shrink_end_speed
         - config.window_shrink_start_speed),
    0.0,
    1.0);
  const auto lerp = [progress](double low_speed, double high_speed) {
    return low_speed + progress * (high_speed - low_speed);
  };

  const double selection_scale =
    lerp(1.0, config.selection_min_window_scale);
  const double firing_scale = lerp(
    config.firing_max_window_scale,
    config.firing_min_window_scale);
  const double hold_margin = config.selection_hold_margin * kDegreeToRad;

  DynamicWindows windows;
  windows.selection = {
    base_enter * selection_scale,
    base_leave * selection_scale};
  windows.hold = {
    windows.selection.enter + hold_margin,
    windows.selection.leave + hold_margin};
  windows.firing = {
    base_enter * firing_scale,
    base_leave * firing_scale};
  return windows;
}

}  // namespace L4Planning
