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

// 切板轨迹规划（sp_vision 2025 readme 4.4 的"方案一·隐式搜索"）。小陀螺切板时
// 射击轨迹出现阶跃，不连续点处速度和加速度无定义，云台强行跟随必然超调或滞后。
// 这里把整条射击轨迹采样出来交给 MPC，让它在加速度上限内提前减速过渡。
//
// enable 为假时整节不生效，Plan 与本节出现之前逐位相同。
struct MpcConfig {
  bool enable{false};
  double dt{0.01};                 // 采样步长，秒
  int horizon{100};                // 采样点数；dt * horizon 是整个窗口长度
  double q_position{9.0e6};        // 位置偏差权重，即 readme 里的"重合度"
  double q_velocity{0.0};          // 速度偏差权重
  double r_input{1.0};             // 加速度代价权重
  // ADMM 罚参数。**不是 sp 的 1.0**：在 50 条实车参考上量过，q_position=9e6、
  // rho=1 时 10 次迭代的中点最差偏离最优解 3.7 度，而最优解本身只偏离参考
  // 0.1 度——那 3.7 度全是没收敛出来的假象。rho=10 同样条件下降到 0.5 度，
  // 25 次迭代降到 0.025 度。再往大反而变慢（100 时 25 次迭代是 0.45 度）。
  double rho{10.0};
  int max_iterations{25};          // 每次求解的迭代上限
  // 收敛判据，单位 rad/s^2。没收敛的解不采用，本帧退回不整形。
  double tolerance{1.0e-3};
  // yaw 与 pitch 的机械能力不同，分开配。这两个数必须来自机械/电控的实测，
  // 现在的取值继承自 sp_vision，同行区间是 40~50 / 50~100。
  double max_yaw_acceleration{50.0};
  double max_pitch_acceleration{100.0};
};

// 装甲板规划器的完整配置：共用的命中解算参数 + 装甲板专有的选板参数。
struct ArmorPlanConfig {
  PlanConfig impact;
  SelectorConfig selector;
  BallisticConfig ballistic;
  MpcConfig mpc;
};

}  // namespace L4Planning
