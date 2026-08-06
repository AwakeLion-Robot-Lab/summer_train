#pragma once

#include "l4_planning/types.hpp"

namespace L4Planning {

// 组装五段延迟。拆分本身在 types.hpp 的 Delay 里，这里只负责把"可测量"
// 和"需标定"两类来源填进去，并明确哪些还是空的。
//
// 可见性分级（沿用 docs/pure_cpp_auto_aim_route.md 的约定）：
//   image_to_plan   本帧曝光时刻到规划时刻，直接可测
//   plan_to_send    规划到下发，直接可测
//   send_to_control 通信 + 电控响应，需标定，来自 PlanConfig
//   control_to_fire 电控收到指令到实际击发，需标定，来自 PlanConfig
//   fire_to_hit     飞行时间，由弹道解算填入
class LatencyCompensator {
public:
  explicit LatencyCompensator(PlanConfig config = {});

  // 用曝光时刻和规划时刻填可测量的段，标定段从配置取。未标定的段留 0，
  // 并由 PlanConfig::fireDelayReady() 告诉 L5 不能解锁开火——这里绝不
  // 用猜测值填补，否则火控门禁会被静默绕过。
  [[nodiscard]] Delay measure(TimePoint image_time, TimePoint plan_time) const;

  [[nodiscard]] const PlanConfig& config() const noexcept { return config_; }

private:
  PlanConfig config_;
};

}  // namespace L4Planning
