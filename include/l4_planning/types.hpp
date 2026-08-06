#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <numbers>
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

// 选板策略参数。角度一律用弧度存储，YAML 侧再做度数换算。
struct SelectorConfig {
  // 低于该角速度按"静止/低速"处理，只挑正对枪口的板；超过则进入反陀螺
  // 策略。单位 radian/second。
  double spin_threshold{2.0};

  // 低速档：法线夹角超过该值的板不可能被打中，直接排除。
  double max_face_angle{60.0 * std::numbers::pi / 180.0};

  // 反陀螺档：只考虑进入视野的一侧。coming 是候选窗口的半宽，leaving 是
  // 即将转走那一侧的截止线——板转过这条线后打过去就已经背对枪口了。
  double coming_angle{40.0 * std::numbers::pi / 180.0};
  double leaving_angle{15.0 * std::numbers::pi / 180.0};

  // 前哨站转速固定且只有 3 块板，窗口比普通车辆宽。
  double outpost_coming_angle{70.0 * std::numbers::pi / 180.0};
  double outpost_leaving_angle{30.0 * std::numbers::pi / 180.0};

  // 锁定迟滞：已锁定的板需要比竞争者差过这个角度才允许换板。
  double switch_hysteresis{5.0 * std::numbers::pi / 180.0};
};

struct PlanConfig {
  int max_iterations{10};
  std::chrono::microseconds fly_time_tolerance{100};
  double switch_dead_zone{5.0};  // degree

  // 弹速缺失或明显异常时使用的兜底初速，单位 m/s。裁判系统上电初期会
  // 回传 0，此时用兜底值仍可解算，但 L5 会因 BadBulletSpeed 拒绝开火。
  double fallback_bullet_speed{23.0};
  double min_valid_bullet_speed{10.0};

  // 线性空气阻力系数，0 表示真空模型。实测标定值约 0.01~0.03。
  double drag_coefficient{0.0};

  SelectorConfig selector;

  // 云台角加速度上限，单位 radian/second²。定点规划器不使用；五次多项式
  // 靠它决定过渡段时长（逐步增大直到峰值加速度落在限内），MPC 用作硬约
  // 束。实车标定前保持空值，轨迹类规划器必须据此退化成定点输出。
  std::optional<double> max_yaw_acceleration;
  std::optional<double> max_pitch_acceleration;

  [[nodiscard]] bool trajectoryLimitsReady() const noexcept
  {
    return max_yaw_acceleration.has_value() && max_pitch_acceleration.has_value();
  }

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
