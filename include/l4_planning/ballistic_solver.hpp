#pragma once

#include "l4_planning/ballistic_model.hpp"
#include "l4_planning/types.hpp"

#include <memory>

namespace L4Planning {

// 弹道**反解**：给定目标点求发射角和飞行时间。输入是**枪管系**下的目标点：
// world 系原点即枪管原点（PnpSolver 用 t_camera_barrel 平移后只做纯旋转到
// world），所以 d = hypot(x, y)、h = z 直接可用，不需要再加枪口偏置。
//
// 求解器与物理模型解耦（见 ballistic_model.hpp）。真空模型有闭式解直接返回；
// 有阻力时用 talos DirectSolver 的高度补偿迭代：从视线角出发，把实际落点与
// 目标的高度差累加回瞄准高度，重新求角，直到落差小于门限。这一迭代天然收敛
// 到低弧，因为起点就在低弧一侧。
//
// 这里**不做**弹速合理性判断。低于多少 m/s 算异常是业务门限，归
// PlanConfig::min_valid_bullet_speed 管；求解器只拒绝数学上无解的输入。
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
  // 真空闭式解，同时用作有阻力时的迭代初值。
  [[nodiscard]] std::optional<double> vacuumPitch(
    double d, double h, double bullet_speed) const;

  BallisticConfig config_;
  std::shared_ptr<const IBallisticModel> model_;
};

}  // namespace L4Planning
