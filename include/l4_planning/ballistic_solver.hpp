#pragma once

#include "l4_planning/types.hpp"

#include <optional>

namespace L4Planning {

// 弹道反解：给定**枪管系**下的目标点求发射角和飞行时间。world 系原点就是枪管
// 原点（PnpSolver 只做平移到枪管再纯旋转到 world），所以 d = hypot(x, y)、
// h = z 可以直接用，不需要再补枪口偏置。
//
// 真空和水平二次阻力共用同一个闭式反解：把 tanθ 当未知量的一元二次方程
//
//   h = A·tanθ - (g·A²)/(2·v0²)·sec²θ
//     ⟹ B·u² - A·u + (B + h) = 0,   u = tanθ,  B = g·A²/(2·v0²)
//
// 其中 A 是**等效射程**：真空时 A = d，有阻力时 A = (e^{k·d} - 1)/k。两种模型
// 的差别只在 A，所以只有一份解析式，drag_coefficient = 0 时严格退化为真空。
//
// 判别式为负表示这个初速打不到这个距离和高度，属于正常的物理无解。两根分别是
// 高抛和低抛，取飞行时间短的低弧：目标外推误差随飞行时间线性放大。
//
// 这里**不判断**弹速是否合理，那是业务门限，归 PlanConfig 管。
[[nodiscard]] std::optional<Ballistic> solveBallistic(
  double distance, double height, double bullet_speed, const BallisticConfig& config);

}  // namespace L4Planning
