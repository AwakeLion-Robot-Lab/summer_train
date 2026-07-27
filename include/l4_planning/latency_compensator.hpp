#pragma once

#include "l4_planning/types.hpp"

#include <cmath>

namespace L4Planning {

struct LatencyConfig {
  // 命令发布到弹丸离开枪口的标定延迟，单位为秒。
  double fire_delay{0.0};

  [[nodiscard]] bool ready() const noexcept
  {
    return std::isfinite(fire_delay) && fire_delay >= 0.0;
  }
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

  // 兼容需要为单次计算显式指定出膛延迟的调用。
  [[nodiscard]] LatencyResult calculate(const Delay& delay) const noexcept;

private:
  LatencyConfig config_;
};

}  // namespace L4Planning
