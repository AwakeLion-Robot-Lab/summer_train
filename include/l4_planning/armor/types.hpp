#pragma once

#include "l4_planning/ballistic.hpp"
#include "l4_planning/types.hpp"

namespace L4Planning {

struct SelectorConfig {
  double coming_angle{60.0 / 57.3};  // 候选板进入可击打区域的角度
  double leaving_angle{20.0 / 57.3}; // 结合旋转方向排除即将离开的板
  // 前哨站转速固定且板面更窄，进入角放宽、离开角收紧，与普通车分开配。
  double outpost_coming_angle{70.0 / 57.3};
  double outpost_leaving_angle{30.0 / 57.3};
};

// 装甲板规划器的完整配置：共用的命中解算参数 + 装甲板专有的选板参数。
struct ArmorPlanConfig {
  PlanConfig impact;
  SelectorConfig selector;
  BallisticConfig ballistic;
};

}  // namespace L4Planning
