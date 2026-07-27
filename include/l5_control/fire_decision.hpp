#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/reject_reason.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace L5Control {

using TimePoint = std::chrono::steady_clock::time_point;

struct FireConfig {
  bool shoot_enable{false};

  std::optional<double> bullet_diameter;
  std::optional<double> min_bullet_speed;
  std::optional<double> max_bullet_speed;
  std::optional<double> heat_limit;

  std::optional<double> min_yaw;
  std::optional<double> max_yaw;
  std::optional<double> min_pitch;
  std::optional<double> max_pitch;
  std::optional<double> yaw_distance_boundary;
  std::optional<double> near_max_yaw_command_jump;
  std::optional<double> near_max_yaw_error;
  std::optional<double> far_max_yaw_command_jump;
  std::optional<double> far_max_yaw_error;
  std::optional<double> max_aim_pitch_error;
  std::optional<double> max_pitch_command_jump;
  std::optional<double> max_pitch_error;

  std::chrono::milliseconds max_robot_state_age{50};
  std::chrono::milliseconds max_gimbal_pose_age{20};
  std::chrono::milliseconds max_plan_age{30};

  [[nodiscard]] bool parametersReady() const noexcept
  {
    const auto finite = [](const std::optional<double>& value) {
      return value.has_value() && std::isfinite(*value);
    };

    return finite(bullet_diameter) && *bullet_diameter > 0.0 &&
           finite(min_bullet_speed) && finite(max_bullet_speed) &&
           *min_bullet_speed > 0.0 && *min_bullet_speed < *max_bullet_speed &&
           finite(heat_limit) && *heat_limit > 0.0 &&
           finite(min_yaw) && finite(max_yaw) && *min_yaw < *max_yaw &&
           finite(min_pitch) && finite(max_pitch) && *min_pitch < *max_pitch &&
           finite(yaw_distance_boundary) && *yaw_distance_boundary > 0.0 &&
           finite(near_max_yaw_command_jump) &&
           *near_max_yaw_command_jump > 0.0 &&
           finite(near_max_yaw_error) && *near_max_yaw_error > 0.0 &&
           finite(far_max_yaw_command_jump) &&
           *far_max_yaw_command_jump > 0.0 &&
           finite(far_max_yaw_error) && *far_max_yaw_error > 0.0 &&
           finite(max_aim_pitch_error) && *max_aim_pitch_error > 0.0 &&
           finite(max_pitch_command_jump) && *max_pitch_command_jump > 0.0 &&
           finite(max_pitch_error) && *max_pitch_error > 0.0 &&
           max_robot_state_age.count() > 0 &&
           max_gimbal_pose_age.count() > 0 &&
           max_plan_age.count() > 0;
  }
};

FireConfig loadFireConfig(const std::string& config_path);

struct FireInput {
  std::optional<L3Estimation::TargetState> target;
  L4Planning::AimPlan plan;
  L1Sensor::RobotState robot_state;

  TimePoint now{};
  double actual_yaw{0.0};
  double actual_pitch{0.0};

  bool calibration_ready{false};
  bool serial_fresh{false};
  bool gimbal_pose_fresh{false};
  bool armor_switching{false};
  bool command_jump{false};
};

struct FireDecision {
  bool fire_feasible{false};
  bool shoot{false};
  std::vector<RejectReason> reasons;
};

class FireEvaluator {
public:
  explicit FireEvaluator(FireConfig config = {});
  [[nodiscard]] FireDecision evaluate(const FireInput& input);

  void reset() noexcept;

private:
  FireConfig config_;
  int last_target_id_{-1};
  int last_armor_id_{-1};
  std::optional<double> last_command_yaw_;
  std::optional<double> last_command_pitch_;
  std::optional<double> last_fly_time_;
  std::size_t stable_tracking_frames_{0};
  TimePoint last_switch_time_{};
};

[[nodiscard]] bool evaluateFire(const L4Planning::AimPlan& plan);
[[nodiscard]] bool shouldFire(const L4Planning::AimPlan& plan);

}  // namespace L5Control
