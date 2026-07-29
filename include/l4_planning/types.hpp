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

// 装甲板选择器的跨帧状态。Unlocked 不属于切换过程，其余四项对应
// 当前锁定、预切换、切换中和新装甲板稳定确认。
enum class ArmorTrackingPhase : std::uint8_t {
  Unlocked,
  Tracking,
  PreSwitch,
  Switching,
  Stabilizing
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

// 非 MPC 模式的瞄准参考。
struct AimReference {
  int target_id{-1};
  int armor_id{-1};
  TimePoint impact_time{};
  bool tracking{false};

  Eigen::Vector3d aim_point_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d aim_point_world{Eigen::Vector3d::Zero()};

  double yaw{0.0};
  double pitch{0.0};
  double yaw_rate{0.0};
  double pitch_rate{0.0};
  double yaw_acceleration{0.0};
  double pitch_acceleration{0.0};

  double fly_time{0.0};
};

// MPC 输出的单个可执行控制点。
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
  int iteration_count{0};       // 最终候选的固定点迭代次数
  double fly_time_error{0.0};   // 最终相邻飞行时间误差，s
  double position_error{0.0};   // 最终相邻装甲板位置误差，m
  double angle_error{0.0};      // 最终相邻 yaw/pitch 合成误差，rad
  bool converged{false};        // 是否满足配置中的收敛条件
};

// AimReference 作为基类，使现有 L5 无需修改即可继续访问 plan.yaw/pitch。
struct Plan : AimReference {
  TimePoint generated_at{};

  // using_MPC=true 时非空，并按 execute_time 排列；首项是当前控制点。
  std::vector<AimSample> samples;

  bool using_MPC{false};       // false：直接参考；true：已生成 MPC samples
  ArmorTrackingPhase tracking_phase{ArmorTrackingPhase::Unlocked};
  bool armor_switching{false}; // 切换中或等待新装甲板稳定
  bool ballistic_valid{false}; // 最终装甲板是否存在有效弹道
  bool fire_permitted{false};  // 规划层是否允许进入射击窗口
  bool valid{false};           // 规划结果是否有效
};

struct PlanConfig {
  int max_iterations{20}; // 单块装甲板最大固定点迭代次数
  std::chrono::microseconds fly_time_tolerance{200}; // 飞行时间收敛阈值
  double position_tolerance{0.005};  // meter
  double angle_tolerance{0.0};       // rad，0 表示暂不启用

  double gravity{9.80665}; // 重力加速度绝对值，单位 m/s^2
  // 弹道模型选择。关闭时使用真空弹道；开启时使用 a_drag=-k*v。
  bool enable_air_resistance{false};
  // 有效线性阻力系数 k，单位 s^-1，需要通过实弹落点标定。
  double linear_drag_coefficient{0.0};

  double switch_dead_zone{5.0};  // degree
  double rotation_rate_dead_zone{0.05}; // rad/s，低于该值按近似静止处理
  int lock_stable_frames{3}; // 新装甲板连续稳定的确认帧数
  // Q_aim_cost 的平滑归一化区间，使用当前云台姿态到候选弹道角的合成角差。
  double aim_cost_good_angle{5.0};  // degree，小于该角度时评分为 1
  double aim_cost_bad_angle{30.0};  // degree，大于该角度时评分为 0

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
