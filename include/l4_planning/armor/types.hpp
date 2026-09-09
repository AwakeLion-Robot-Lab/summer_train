#pragma once

#include "l4_planning/aim_smoother.hpp"
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

// 切板过渡段：sp_vision readme 4.4 的"方案二·显式搜索"。小陀螺切板时射击
// 轨迹出现阶跃，不连续点处速度和加速度无定义，云台强行跟随必然超调或滞后；
// 在切板前插一段五次多项式，使规划后轨迹的加速度不超过云台能力上限。
struct BlendConfig {
  // 默认关。这是实验特性，而且 limits 里的加速度上限没有实车标定之前，
  // 过渡时长等于随手填的——那还不如不做。
  bool enable{false};

  BlendLimits limits;

  // 前视窗口，s。必须大于 limits.max_duration，否则发现切板时已经来不及
  // 提前减速，每次都只能走 late 分支。
  double horizon{0.240};

  // 前视扫描步长，s。切板时刻的分辨率就是它，同时决定每帧多做几次整车外推。
  double grid{0.004};

  // 数值求导步长，s。太小会被 EKF 外推自身的数值噪声吃掉，太大则把加速度
  // 抹平——加速度正是约束的输入。
  double derivative_step{0.002};
};

// 装甲板规划器的完整配置：共用的命中解算参数 + 装甲板专有的选板参数。
struct ArmorPlanConfig {
  PlanConfig impact;
  SelectorConfig selector;
  BallisticConfig ballistic;
  BlendConfig blend;
};

}  // namespace L4Planning
