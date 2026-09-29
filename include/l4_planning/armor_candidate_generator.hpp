#pragma once

#include "l4_planning/state.hpp"

#include <optional>
#include <vector>

namespace L1Sensor {
struct RobotState;
}

namespace L3Estimation {
struct TargetState;
}

namespace L4Planning {

struct PlannerContext;

// 为目标的每块装甲板独立执行命中时刻固定点迭代、弹道求解和质量评分。
// nullopt 表示连发射时刻的整车预测都无效，调用方应保持本周期规划失败。
[[nodiscard]] std::optional<std::vector<ArmorCandidate>>
generateArmorCandidates(
  const L3Estimation::TargetState& target_state,
  const L1Sensor::RobotState& robot_state,
  const PlannerContext& context,
  TimePoint fire_time);

}  // namespace L4Planning
