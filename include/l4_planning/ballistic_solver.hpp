#pragma once

#include "l4_planning/types.hpp"

namespace L4Planning {

// 弹道解算。输入是**枪管系**下的目标点：world 系原点即枪管原点
// （PnpSolver 用 t_camera_barrel 平移后只做纯旋转到 world），所以
// d = hypot(x, y)、h = z 直接可用，不需要再加枪口偏置。
//
// 当前实现是真空模型的闭式解。drag_coefficient 预留给线性空气阻力
// dv/dt = -k*v - g；k 为 0 时严格退化成真空解，因此接口不必改动即可
// 平滑升级（参考 Climber_Vision 的 AirResistTrajectory）。
class BallisticSolver {
public:
  explicit BallisticSolver(double drag_coefficient = 0.0);

  // d 水平距离 (m)，h 竖直高度 (m，向上为正)，bullet_speed 初速 (m/s)。
  // 无解时返回 valid = false，调用方不得使用 pitch / fly_time。
  [[nodiscard]] Ballistic solve(double d, double h, double bullet_speed) const;

  // 兼容旧接口：只取俯仰角，无解时返回 0。新代码应当直接用 solve()。
  [[nodiscard]] double solvePitch(double distance, double height, double bullet_speed) const;

private:
  double drag_coefficient_{0.0};
};

}  // namespace L4Planning
