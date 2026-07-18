#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/reject_reason.hpp"

#include <chrono>
#include <optional>
#include <vector>

namespace L5Control {

struct FireConfig {
  // 第一阶段必须保持 false；完成验收后由实车配置显式修改。
  bool shoot_enable{false};

  std::optional<double> bullet_diameter;  // meter; 17 mm projectile = 0.017
  std::optional<double> min_bullet_speed;  // meter per second
  std::optional<double> max_bullet_speed;  // meter per second
  std::optional<double> heat_limit;

  std::optional<double> min_yaw;
  std::optional<double> max_yaw;
  std::optional<double> min_pitch;
  std::optional<double> max_pitch;

  std::chrono::milliseconds max_robot_state_age{50};
  std::chrono::milliseconds max_gimbal_pose_age{20};

  [[nodiscard]] bool parametersReady() const noexcept
  {
    return bullet_diameter.has_value() && min_bullet_speed.has_value() &&
           max_bullet_speed.has_value() && heat_limit.has_value() &&
           min_yaw.has_value() && max_yaw.has_value() &&
           min_pitch.has_value() && max_pitch.has_value();
  }
};

struct FireInput {
  std::optional<L3Estimation::Target> target;
  L4Planning::Plan plan;
  L1Sensor::RobotState robot_state;

  std::chrono::steady_clock::time_point now{};
  double actual_yaw{0.0};
  double actual_pitch{0.0};

  bool calibration_ready{false};
  bool serial_fresh{false};
  bool gimbal_pose_fresh{false};
  bool armor_switching{false};
  bool command_jump{false};
};

struct FireDecision {
  // fire_feasible 记录理论窗口；shoot 是考虑 shoot_enable 后的实际下发值。
  bool fire_feasible{false};
  bool shoot{false};
  std::vector<RejectReason> reasons;
};

bool shouldFire(const L4Planning::AimPlan& plan);

}  // namespace L5Control
