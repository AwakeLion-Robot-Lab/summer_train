#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <string_view>

namespace L4Planning {

// 弹道**模型**：描述物理，不描述数值方法。
//
//   impact()  正向：给定发射角，算出子弹飞到指定水平距离时的高度和用时。
//   launch()  反向：给定目标点，算出发射角和飞行时间。**只有存在闭式解的模型
//             才覆盖它**，默认返回 nullopt，BallisticSolver 会退化成高度补偿
//             迭代。这样将来加一个没有闭式解的模型（比如沿速度方向的全二次
//             阻力 + RK4 积分，jlu_vision_26 的做法）不必改求解器一行。
//
// 正向/反向拆分抄自 talos 的 core/trajectory；把反向也放进模型是 newvision
// 的改动，理由见 ballistic_solver.hpp。
struct Impact {
  double z{0.0};         // 子弹到达该水平距离时的高度，向上为正，单位 m
  double fly_time{0.0};  // 飞行时间，单位 s
};

struct Launch {
  double pitch{0.0};     // 发射角，抬头为正，单位 rad
  double fly_time{0.0};  // 飞行时间，单位 s
};

// 真空和水平二次阻力共用同一个反解：把 tanθ 当未知量的一元二次方程
//
//   h = A·tanθ - (g·A²)/(2·v0²)·sec²θ
//     ⟹ B·u² - A·u + (B + h) = 0,   u = tanθ,  B = g·A²/(2·v0²)
//
// 其中 A 是**等效射程**：真空时 A = d，有阻力时 A = (e^{k·d} - 1)/k。
// 两种模型的差别**只在 A**，所以反解只写一份。
//
// 判别式为负表示该初速打不到这个距离和高度（目标落在安全抛物面之外），
// 属于正常的物理无解。两根分别对应高抛和低抛，取飞行时间短的低弧：目标外推
// 误差随飞行时间线性放大，而且高抛更容易撞到场地限高。
[[nodiscard]] inline std::optional<Launch> solveByEffectiveRange(
  double effective_range, double height, double v0, double gravity) noexcept
{
  const double a = effective_range;
  if (!std::isfinite(a) || a <= 0.0 || v0 < 1e-6) {
    return std::nullopt;
  }

  const double b = gravity * a * a / (2.0 * v0 * v0);
  if (b < 1e-12) {
    return std::nullopt;
  }

  const double discriminant = a * a - 4.0 * b * (b + height);
  if (!(discriminant >= 0.0)) {
    return std::nullopt;
  }

  const double sqrt_discriminant = std::sqrt(discriminant);
  const double pitch_low = std::atan((a - sqrt_discriminant) / (2.0 * b));
  const double pitch_high = std::atan((a + sqrt_discriminant) / (2.0 * b));

  const auto flight_time = [a, v0](double pitch) {
    const double horizontal = v0 * std::cos(pitch);
    return horizontal > 1e-6 ? a / horizontal
                             : std::numeric_limits<double>::infinity();
  };

  const double time_low = flight_time(pitch_low);
  const double time_high = flight_time(pitch_high);
  const bool take_low = time_low <= time_high;
  const double pitch = take_low ? pitch_low : pitch_high;
  const double fly_time = take_low ? time_low : time_high;

  if (!std::isfinite(pitch) || !std::isfinite(fly_time) || fly_time <= 0.0) {
    return std::nullopt;
  }
  return Launch{pitch, fly_time};
}

class IBallisticModel {
public:
  virtual ~IBallisticModel() = default;

  IBallisticModel() = default;
  IBallisticModel(const IBallisticModel&) = delete;
  IBallisticModel& operator=(const IBallisticModel&) = delete;
  IBallisticModel(IBallisticModel&&) = delete;
  IBallisticModel& operator=(IBallisticModel&&) = delete;

  // range 水平距离 (m)，pitch 发射角 (rad，抬头为正)，v0 初速 (m/s)。
  // 该角度打不到这个水平距离时返回 nullopt。
  [[nodiscard]] virtual std::optional<Impact> impact(
    double range, double pitch, double v0) const noexcept = 0;

  // 闭式反解。没有闭式解的模型保持默认实现，求解器据此退化成迭代。
  [[nodiscard]] virtual std::optional<Launch> launch(
    double range, double height, double v0) const noexcept
  {
    (void)range;
    (void)height;
    (void)v0;
    return std::nullopt;
  }

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

// 真空抛体。x = v0·cosθ·t，z = v0·sinθ·t - g·t²/2。
class VacuumModel final : public IBallisticModel {
public:
  explicit VacuumModel(double gravity = 9.7833) noexcept : gravity_(gravity) {}

  [[nodiscard]] std::optional<Impact> impact(
    double range, double pitch, double v0) const noexcept override
  {
    const double cos_pitch = std::cos(pitch);
    if (v0 < 1e-6 || range < 0.0 || std::abs(cos_pitch) < 1e-6) {
      return std::nullopt;
    }

    const double fly_time = range / (v0 * cos_pitch);
    if (fly_time < 0.0 || !std::isfinite(fly_time)) {
      return std::nullopt;
    }

    const double z =
      v0 * std::sin(pitch) * fly_time - 0.5 * gravity_ * fly_time * fly_time;
    return Impact{z, fly_time};
  }

  // 等效射程就是水平距离本身。
  [[nodiscard]] std::optional<Launch> launch(
    double range, double height, double v0) const noexcept override
  {
    return solveByEffectiveRange(range, height, v0, gravity_);
  }

  [[nodiscard]] std::string_view name() const noexcept override { return "vacuum"; }
  [[nodiscard]] double gravity() const noexcept { return gravity_; }

private:
  double gravity_;
};

// **二次**空气阻力，只作用在水平分量上：
//
//   dvx/dt = -k·vx²     （等价于 dvx/dx = -k·vx，即 vx(x) = v0·cosθ·e^{-k·x}）
//   dvz/dt = -g         （竖直方向仍为自由落体）
//
//   t(x) = (e^{k·x} - 1) / (k·v0·cosθ)
//   z(t) = v0·sinθ·t - g·t²/2
//
// 阻力**是二次的，不是线性的**。这一点很容易搞错——FYT2024_vision 的
// ResistanceCompensator、talos 的 LinearDragModel 都用同一个公式却都命名/注释
// 成 "linear"。判据有两个：
//
//  1. 对 t(x) 求导得 vx = v0·cosθ·e^{-k·x}，回代 dvx/dt = vx·dvx/dx = -k·vx²；
//  2. 量纲。k 出现在 e^{k·range} 里，所以 k 的单位必须是 1/m；线性阻力系数
//     的单位是 1/s。**Climber_Vision 的 air_resistance_k 是 1/s 的线性系数，
//     数值不能搬过来**，两边的 0.02 完全不是一回事。
//
// 二次阻力才是 17mm 弹丸的正确物理：v≈23 m/s 时 Re≈3×10⁴，远在二次阻力区。
// k = ρ·C_d·A/(2m)，按 3.2 g、C_d≈0.45 算出约 0.019 /m，与 jlu_vision_26 标定
// 的 0.01903 吻合。
//
// 相比 Climber_Vision 在两个方向都加阻力的模型，这里牺牲了竖直方向的阻力项，
// 换来正向和反向都有闭式解——17mm 弹丸在 10 m 内竖直速度远小于水平速度，这一
// 项的影响比水平衰减小一个量级。
class QuadraticDragModel final : public IBallisticModel {
public:
  QuadraticDragModel(double gravity, double drag_coefficient) noexcept
  : gravity_(gravity), drag_(drag_coefficient)
  {
  }

  [[nodiscard]] std::optional<Impact> impact(
    double range, double pitch, double v0) const noexcept override
  {
    const double cos_pitch = std::cos(pitch);
    if (v0 < 1e-6 || range < 0.0 || std::abs(cos_pitch) < 1e-6) {
      return std::nullopt;
    }

    const double fly_time = effectiveRange(range) / (v0 * cos_pitch);
    if (fly_time < 0.0 || !std::isfinite(fly_time)) {
      return std::nullopt;
    }

    const double z =
      v0 * std::sin(pitch) * fly_time - 0.5 * gravity_ * fly_time * fly_time;
    return Impact{z, fly_time};
  }

  // 反解是**精确闭式解**，不需要迭代：把等效射程代进同一个 tanθ 二次方程即可。
  // 这一步抄自 awakening 的 BallisticTrajectory::solve_pitch —— 工作空间里只有
  // 它推出了这个式子，FYT / talos 都在对同一个模型做没必要的高度补偿迭代。
  [[nodiscard]] std::optional<Launch> launch(
    double range, double height, double v0) const noexcept override
  {
    return solveByEffectiveRange(effectiveRange(range), height, v0, gravity_);
  }

  [[nodiscard]] std::string_view name() const noexcept override
  {
    return "quadratic_drag";
  }
  [[nodiscard]] double gravity() const noexcept { return gravity_; }
  [[nodiscard]] double drag() const noexcept { return drag_; }

private:
  // 等效射程 A = (e^{k·d} - 1)/k。k 趋于 0 时是 0/0，用真空解的极限收尾，
  // 保证 k = 0 严格退化成真空模型。
  [[nodiscard]] double effectiveRange(double range) const noexcept
  {
    return drag_ < 1e-6 ? range : std::expm1(drag_ * range) / drag_;
  }

  double gravity_;
  double drag_;
};

}  // namespace L4Planning
