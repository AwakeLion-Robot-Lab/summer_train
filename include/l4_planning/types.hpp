#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <vector>

namespace L4Planning {

using TimePoint = std::chrono::steady_clock::time_point;

enum class PlanType : std::uint8_t {
  Direct,
  Setpoint,
  QuinticSwitch,
  TinyMpc
};

enum class AimPlanStatus : std::uint8_t {
  NoTarget,
  Tracking,
  Switching,
  Ready,
  Failed
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

// 一次规划实际使用的延迟结果，单位统一为秒。
  struct Delay {
    TimePoint camera_timestamp{};
    TimePoint command_timestamp{};
    double fire_delay{0.0};

    [[nodiscard]] double beforeFire() const noexcept
    {
      return std::chrono::duration<double>(
        command_timestamp - camera_timestamp).count();
    }

    [[nodiscard]] double total() const noexcept
    {
      return beforeFire() + fire_delay;
    }
  };

struct Ballistic {
  double yaw{0.0};
  double pitch{0.0};
  double fly_time{0.0};
  bool valid{false};
};

struct AimReferenceSample {
  TimePoint command_time{};
  TimePoint fire_time{};
  TimePoint impact_time{};

  double yaw{0.0};
  double pitch{0.0};
  double yaw_rate{0.0};
  double pitch_rate{0.0};
  double yaw_acceleration{0.0};
  double pitch_acceleration{0.0};

  double fly_time{0.0};
  double confidence{0.0};
  int target_id{-1};
  int armor_id{-1};
  Eigen::Vector3d aim_point_world{Eigen::Vector3d::Zero()};
  bool valid{false};
};

struct AimSample {
  TimePoint execute_time{};

  double yaw{0.0};
  double pitch{0.0};
  double yaw_rate{0.0};
  double pitch_rate{0.0};
  double yaw_acceleration{0.0};
  double pitch_acceleration{0.0};
  double yaw_jerk{0.0};
  double pitch_jerk{0.0};
};

struct PlanningDiagnostics {
  int iteration_count{0};
  double fly_time_error{0.0};
  double position_error{0.0};
  double angle_error{0.0};
  bool converged{false};
};

// L4 的完整输出。第一版 Setpoint 规划器将速度和加速度保持为 0。
struct Plan {
  std::uint64_t sequence{0};
  int target_id{-1};
  int armor_id{-1};
  int selected_armor_id{-1};

  TimePoint plan_time{};
  TimePoint generated_at{};
  TimePoint valid_until{};
  TimePoint fire_time{};
  TimePoint hit_time{};

  Eigen::Vector3d aim_point{Eigen::Vector3d::Zero()};  // barrel frame, meter

  double yaw{0.0};
  double pitch{0.0};
  double yaw_vel{0.0};
  double pitch_vel{0.0};
  double yaw_acc{0.0};
  double pitch_acc{0.0};

  double fly_time{0.0};
  Delay delay;
  std::vector<AimSample> samples;
  std::vector<AimReferenceSample> reference;
  PlanningDiagnostics diagnostics;
  double confidence{0.0};

  AimPlanStatus status{AimPlanStatus::NoTarget};
  PlanType planner_type{PlanType::Direct};
  bool ballistic_valid{false};
  bool fire_permitted{false};

  PlanType type{PlanType::Setpoint};
  PlanError error{PlanError::NoTarget};
  bool valid{false};
};

struct PlanConfig {
  int max_iterations{20};
  std::chrono::microseconds fly_time_tolerance{200};
  double position_tolerance{0.005};  // meter
  double angle_tolerance{0.0};       // rad，0 表示暂不启用
  double switch_dead_zone{5.0};  // degree

  double normal_enter_angle{60.0};   // degree
  double normal_leave_angle{20.0};   // degree
  double outpost_enter_angle{70.0};  // degree
  double outpost_leave_angle{30.0};  // degree
  int max_lost_frames{5};

  double yaw_angle_weight{9000000.0};
  double yaw_velocity_weight{0.0};
  double yaw_acceleration_weight{1.0};
  double pitch_angle_weight{9000000.0};
  double pitch_velocity_weight{0.0};
  double pitch_acceleration_weight{1.0};
  double min_yaw_acceleration{-50.0};
  double max_yaw_acceleration{50.0};
  double min_pitch_acceleration{-100.0};
  double max_pitch_acceleration{100.0};

};

// 兼容当前代码中已经使用的名称。
using AimPlan = Plan;
using DelayBreakdown = Delay;
using BallisticResult = Ballistic;
using PlannerType = PlanType;
using PlannerConfig = PlanConfig;
using PlanRejectReason = PlanError;

}  // namespace L4Planning
