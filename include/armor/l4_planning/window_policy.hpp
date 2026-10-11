#pragma once

#include "l4_planning/types.hpp"

namespace L4Planning {

// 以弧度表示的单侧进入角和离开角；完整窗口为 [-enter, leave]。
struct AngularWindow {
  double enter{0.0};
  double leave{0.0};
};

// 选板、当前板保持和开火使用相互嵌套的三个窗口。
struct DynamicWindows {
  AngularWindow selection;
  AngularWindow hold;
  AngularWindow firing;
};

// 根据装甲板相对视线角速度作分段线性缩放。动态策略关闭时三个窗口
// 都退化为原始窗口，保持旧配置的行为。
[[nodiscard]] DynamicWindows computeDynamicWindows(
  double base_enter_angle_degree,
  double base_leave_angle_degree,
  double relative_yaw_rate,
  const PlannerConfig& config) noexcept;

}  // namespace L4Planning
