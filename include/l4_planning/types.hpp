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
  // 延迟链还没在实车上标定完，只跟随不开火。
  DelayNotCalibrated,
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

  double beforeFire() const noexcept
  {
    return image_to_plan + plan_to_send + send_to_control + control_to_fire;
  }

  double total() const noexcept
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
  // 命中目标面在各自规划器里的编号：装甲板是整车展开后的物理板编号，
  // 符是叶片编号。字段名沿用 armor_ 前缀只是历史包袱，不是类型依赖——
  // 共享层不引用任何 armor/ 头文件。等 buff 规划器落地再一起改名。
  int armor_id{-1};
  Eigen::Vector4d armor_pose{Eigen::Vector4d::Zero()};

  Eigen::Vector3d point() const noexcept
  {
    return armor_pose.head<3>();
  }

  double facingAngle() const noexcept
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

  bool valid() const noexcept
  {
    return status != PlanStatus::Rejected;
  }

  bool fireAdmissible() const noexcept
  {
    return status == PlanStatus::FireReady;
  }
};

// 命中解算的共用参数：弹道、延迟链、弹速。与目标是装甲板还是符无关，
// 两条规划链路用同一组。选板一类的目标专有参数放各自的 armor/ 或 buff/。
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

  // 串口发出到电控执行的耗时，只能在实车上标定，未标定时保持空值。
  // 其余四段都是可算或可测的：image_to_plan 和 plan_to_send 由 runtime 实测，
  // control_to_fire 用上面的高低速档，fire_to_hit 是弹道飞行时间。
  std::optional<double> send_to_control;

  bool bulletSpeedValid(double speed) const noexcept
  {
    return std::isfinite(speed) && speed >= min_valid_bullet_speed;
  }

  // 延迟链是否已经完整到可以开火。缺这一段时目标外推的落点会系统性偏早，
  // 所以 Planner 会把计划降级成 TrackOnly：云台照常跟随，但不允许开火。
  bool fireDelayReady() const noexcept
  {
    return send_to_control.has_value();
  }
};

}  // namespace L4Planning
