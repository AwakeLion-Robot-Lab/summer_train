#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <optional>

namespace L4Planning {

using TimePoint = std::chrono::steady_clock::time_point;

enum class PlanType : std::uint8_t {
  Setpoint,
  QuinticSwitch,
  TinyMpc
};

enum class PlanError : std::uint8_t {
  None,
  NoTarget,
  NotTracking,
  NoArmor,
  NoCalibration,
  BadBulletSpeed,
  BallisticFailed,
  NotConverged,
  OutOfRange,
  Switching
};

// 单位统一为秒。拆开保存，禁止在 runtime 中只维护一个含义不清的
// 总延迟。
struct Delay {
  double image_to_plan{0.0};
  double plan_to_send{0.0};
  double send_to_control{0.0};
  double control_to_fire{0.0};
  double fire_to_hit{0.0};

  [[nodiscard]] double beforeFire() const noexcept
  {
    return image_to_plan + plan_to_send + send_to_control + control_to_fire;
  }

  [[nodiscard]] double total() const noexcept
  {
    return beforeFire() + fire_to_hit;
  }
};

struct Ballistic {
  double yaw{0.0};
  double pitch{0.0};
  double fly_time{0.0};
  bool valid{false};
};

// L4 的完整输出。第一版 Setpoint 规划器将速度和加速度保持为 0。
struct Plan {
  int target_id{-1};
  int armor_id{-1};

  TimePoint plan_time{};
  TimePoint fire_time{};
  TimePoint hit_time{};

  Eigen::Vector3d aim_point{Eigen::Vector3d::Zero()};  // muzzle frame, meter

  double yaw{0.0};
  double pitch{0.0};
  double yaw_vel{0.0};
  double pitch_vel{0.0};
  double yaw_acc{0.0};
  double pitch_acc{0.0};

  double fly_time{0.0};
  Delay delay;
  bool ballistic_valid{false};

  PlanType type{PlanType::Setpoint};
  PlanError error{PlanError::NoTarget};
  bool valid{false};
};

struct PlanConfig {
  int max_iterations{10};
  std::chrono::microseconds fly_time_tolerance{100};
  double switch_dead_zone{5.0};  // degree

  // 实车标定前保持空值；空值表示不能解锁开火。
  std::optional<double> send_to_control;  // second
  std::optional<double> control_to_fire;  // second

  [[nodiscard]] bool fireDelayReady() const noexcept
  {
    return send_to_control.has_value() && control_to_fire.has_value();
  }
};

// 兼容当前代码中已经使用的名称。
using AimPlan = Plan;
using DelayBreakdown = Delay;
using BallisticResult = Ballistic;
using PlannerType = PlanType;
using PlanRejectReason = PlanError;

}  // namespace L4Planning
