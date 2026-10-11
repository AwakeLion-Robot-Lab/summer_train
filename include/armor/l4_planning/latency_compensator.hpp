#pragma once

#include "l4_planning/types.hpp"

#include <cmath>

namespace L4Planning {

struct LatencyConfig {
  // 根据目标有符号 yaw 角速度选择的命令发布到弹丸出膛延迟，单位为秒。
  // 与旧 L4 语义一致：只有正向 yaw_rate 超过分界才使用高速档。
  double high_speed_fire_delay{0.030};
  double low_speed_fire_delay{0.015};
  double decision_speed{8.0}; // rad/s

  [[nodiscard]] bool ready() const noexcept
  {
    return std::isfinite(high_speed_fire_delay)
           && high_speed_fire_delay >= 0.0
           && std::isfinite(low_speed_fire_delay)
           && low_speed_fire_delay >= 0.0
           && std::isfinite(decision_speed)
           && decision_speed >= 0.0;
  }

  [[nodiscard]] double fireDelay(double target_yaw_rate) const noexcept;
};

struct LatencyResult {
  Delay delay;
  bool valid{false};
};

class LatencyCompensator {
public:
  explicit LatencyCompensator(LatencyConfig config = {});

  // 使用配置中的出膛延迟，计算图像时刻到实际出膛时刻的总延迟。
  [[nodiscard]] LatencyResult calculate(
    TimePoint camera_timestamp,
    TimePoint command_timestamp) const noexcept;

  // 按目标有符号 yaw 角速度动态选择高、低速出膛延迟。
  [[nodiscard]] LatencyResult calculate(
    TimePoint camera_timestamp,
    TimePoint command_timestamp,
    double target_yaw_rate) const noexcept;

  // 兼容需要为单次计算显式指定出膛延迟的调用。
  [[nodiscard]] LatencyResult calculate(const Delay& delay) const noexcept;

private:
  LatencyConfig config_;
};

}  // namespace L4Planning
