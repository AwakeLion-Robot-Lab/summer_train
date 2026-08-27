#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>

namespace L4Planning {

using TimePoint = std::chrono::steady_clock::time_point;

enum class PlanError : std::uint8_t {
  None,
  NoTarget,
  BadBulletSpeed,
  BallisticFailed,
  OutOfWindow
};

// 一帧观测从曝光到命中的延迟链，所有字段单位均为秒。
struct Delay {
  double image_to_plan{0.0};      // 曝光中点 -> 开始规划
  double plan_to_send{0.0};       // 规划完成 -> 串口发送
  double send_to_control{0.0};    // 串口发送 -> 电控执行
  double control_to_fire{0.0};    // 电控执行 -> 弹丸离膛
  double fire_to_hit{0.0};        // 弹丸离膛 -> 命中目标

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
  double yaw{0.0};       // rad；当前求解器只解 pitch，保持为 0
  double pitch{0.0};     // rad
  double fly_time{0.0};  // s
  bool valid{false};
};

enum class PlanStatus : std::uint8_t {
  Rejected,   // 不产生新的瞄准命令
  TrackOnly,  // 可以跟随，但本帧禁止开火
  FireReady   // L4 允许开火，仍需经过 L5 判定
};

// 命中点及其对应的世界系枪管角命令。
struct AimReference {
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double pitch{0.0};
};

// L5 判定始终落在一块实体装甲板上；armor_pose = [x, y, z, normal_yaw]。
struct FireReference {
  int armor_id{-1};
  Eigen::Vector4d armor_pose{Eigen::Vector4d::Zero()};

  [[nodiscard]] Eigen::Vector3d point() const noexcept
  {
    return armor_pose.head<3>();
  }

  [[nodiscard]] double facingAngle() const noexcept
  {
    // 视线方向减板面法向并归一化到 [-pi, pi]。
    const double line_of_sight = std::atan2(armor_pose.y(), armor_pose.x());
    return std::remainder(
      line_of_sight - armor_pose.w(), 2.0 * std::numbers::pi);
  }
};

struct PlanTiming {
  // 最终瞄准点所对应的目标时刻，当前即预计命中时刻。
  TimePoint prediction_time{};
  double fly_time{0.0};  // s
  Delay delay;
};

struct Plan {
  PlanStatus status{PlanStatus::Rejected};
  PlanError reason{PlanError::NoTarget};
  AimReference aim;
  std::optional<FireReference> fire;
  PlanTiming timing;

  [[nodiscard]] bool valid() const noexcept
  {
    return status != PlanStatus::Rejected;
  }

  [[nodiscard]] bool fireAdmissible() const noexcept
  {
    return status == PlanStatus::FireReady;
  }
};

struct SelectorConfig {
  double coming_angle{60.0 / 57.3};  // 候选板进入可击打区域的角度
  double leaving_angle{20.0 / 57.3}; // 结合旋转方向排除即将离开的板
};

struct PlanConfig {
  // 飞行时间与目标位置相互依赖，迭代到相邻两次飞行时间之差小于该阈值。
  int max_iterations{10};
  std::chrono::microseconds fly_time_tolerance{1000};

  // 根据整车 yaw 角速度选择的发射延迟，单位秒和 rad/s。
  double high_speed_delay_time{0.030};
  double low_speed_delay_time{0.015};
  double decision_speed{8.0};
  double yaw_offset{0.0};
  double pitch_offset{0.0};

  double fallback_bullet_speed{23.0};
  double min_valid_bullet_speed{14.0};

  SelectorConfig selector;

  // 预留的实车延迟标定值；未标定时保持空值。
  std::optional<double> send_to_control;
  std::optional<double> control_to_fire;

  [[nodiscard]] bool bulletSpeedValid(double speed) const noexcept
  {
    return std::isfinite(speed) && speed >= min_valid_bullet_speed;
  }

  [[nodiscard]] bool fireDelayReady() const noexcept
  {
    return send_to_control.has_value() && control_to_fire.has_value();
  }
};

}  // namespace L4Planning
