#pragma once

#include "l4_planning/ballistic_model.hpp"
#include "l4_planning/types.hpp"

#include <memory>

namespace L4Planning {

// 弹道**反解**：给定目标点求发射角和飞行时间。输入是**枪管系**下的目标点：
// world 系原点即枪管原点（PnpSolver 用 t_camera_barrel 平移后只做纯旋转到
// world），所以 d = hypot(x, y)、h = z 直接可用，不需要再加枪口偏置。
//
// 求解策略分两条路，由模型自己决定走哪条：
//
//  1. **模型给得出闭式反解**（真空、水平二次阻力）→ 直接用，一次算完，精确。
//  2. **模型给不出**（将来的全二次阻力 + RK4）→ 退化成 talos DirectSolver 的
//     高度补偿迭代：把实际落点与目标的高度差累加回瞄准高度，重新求角，直到
//     落差小于门限。内层映射用真空闭式解，比 talos 用视线角 atan2 收敛快。
//
// 现在两个内置模型都走第 1 条。保留第 2 条不是为了兼容，而是为了让"加一个没
// 有闭式解的模型"不必改这个文件——迭代路径由 IBallisticModel::launch() 返回
// nullopt 自动触发。
//
// 为什么不像 FYT / talos 那样一律迭代：这个模型有精确闭式解（awakening 推出
// 来了）。迭代版在 height_tolerance 内提前退出，留下系统性偏差；而且迭代次数
// 随 k 和距离增长，k=0.092、d=10 m 时要 12 次，逼近 max_iterations 上限，再远
// 就会直接报无解。
//
// 这里**不做**弹速合理性判断。多少 m/s 算异常是业务门限，归 PlanConfig 的
// min/max_valid_bullet_speed 管；求解器只拒绝数学上无解的输入。
class BallisticSolver {
public:
  explicit BallisticSolver(BallisticConfig config = {});

  // d 水平距离 (m)，h 竖直高度 (m，向上为正)，bullet_speed 初速 (m/s)。
  // 无解时返回 valid = false，调用方不得使用 pitch / fly_time。
  [[nodiscard]] Ballistic solve(double d, double h, double bullet_speed) const;

  // 兼容旧接口：只取俯仰角，无解时返回 0。新代码应当直接用 solve()。
  [[nodiscard]] double solvePitch(double distance, double height, double bullet_speed) const;

  [[nodiscard]] const IBallisticModel& model() const noexcept { return *model_; }
  [[nodiscard]] const BallisticConfig& config() const noexcept { return config_; }

private:
  // 模型没有闭式反解时的兜底：高度补偿迭代。
  [[nodiscard]] Ballistic solveByHeightCompensation(
    double d, double h, double bullet_speed) const;

  BallisticConfig config_;
  std::shared_ptr<const IBallisticModel> model_;
};

}  // namespace L4Planning
