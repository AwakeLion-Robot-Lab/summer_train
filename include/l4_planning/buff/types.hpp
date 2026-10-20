#pragma once

#include "l3_estimation/buff/types.hpp"

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <cmath>

namespace L4Planning::Buff {

using TimePoint = std::chrono::steady_clock::time_point;

struct BulletSpeed {
  double value{22.5};
  // measured=false 表示 value 来自配置回退值，而不是本帧电控上报。
  bool measured{false};
  // trusted 控制是否允许进入开火节拍；未实测的回退值默认只用于继续跟随。
  bool trusted{false};
};

struct Config {
  double fallback_bullet_speed{22.5};
  double min_valid_bullet_speed{14.0};
  bool trust_fallback_bullet_speed{false};
  double shoot_delay{0.04};
  double offset_yaw_deg{0.0};
  double offset_pitch_deg{0.0};
  double idle_duration{0.4};
  double shoot_duration{0.2};

  [[nodiscard]] bool bulletSpeedValid(double speed) const noexcept
  {
    return std::isfinite(speed) && speed >= min_valid_bullet_speed;
  }

  [[nodiscard]] BulletSpeed resolveBulletSpeed(double reported_speed) const noexcept
  {
    if (bulletSpeedValid(reported_speed)) {
      return {.value = reported_speed, .measured = true, .trusted = true};
    }
    return {
      .value = fallback_bullet_speed,
      .measured = false,
      .trusted = trust_fallback_bullet_speed};
  }
};

struct Plan {
  bool valid{false};
  bool degraded{false};
  bool pre_aim{false};
  bool algorithm_shoot{false};
  int blade_index{-1};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d attack{Eigen::Vector3d::Zero()};
  // L4 在命中时刻选中的靶面外接正方形，按连续环绕顺序存储。它只供叠加层
  // 重投影，不参与弹道或开火判定；退化到符中心时保持 invalid。
  std::array<Eigen::Vector3d, 4> attack_box{};
  bool attack_box_valid{false};
  Eigen::Vector3d target_direction{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double pitch{0.0};
  double fly_time{0.0};
  BulletSpeed bullet_speed{};
  TimePoint prediction_time{};
};

}  // namespace L4Planning::Buff
