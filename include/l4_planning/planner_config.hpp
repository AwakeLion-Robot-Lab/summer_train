#pragma once

#include "l4_planning/latency_compensator.hpp"
#include "l4_planning/state.hpp"
#include "l4_planning/types.hpp"

#include <string>

namespace L4Planning {

// 需要现场标定或调参的 L4 配置集合。
struct PlannerTuning {
  PlannerConfig planner;
  LatencyConfig latency;
  ArmorScoreWeights armor_score_weights;
  // 下位机弹速无效时用于弹道跟随和开火判定的配置回退值。
  // 运行时仍会记录正在使用回退值，便于排查下位机弹速。
  double default_bullet_speed{27.0}; // m/s
  double facing_angle_good{5.0}; // degree
  double facing_angle_bad{25.0}; // degree
};

// 从 YAML 加载 L4 人工调节参数。字段缺失时沿用结构体默认值；字段类型
// 错误或组合不合法时抛出异常，避免带着危险参数继续运行。
[[nodiscard]] PlannerTuning loadPlannerTuning(
  const std::string& config_path);

}  // namespace L4Planning
