#pragma once

#include <cmath>
#include <optional>
#include <string_view>

namespace L4Planning {

// 弹道**模型**：给定发射角，正向算出子弹飞到指定水平距离时的高度和用时。
// 求解（反解发射角）是 BallisticSolver 的事。
//
// 这个正向/反向拆分抄自 talos 的 core/trajectory：模型只描述物理，求解器
// 只描述数值方法，换模型不必动求解器。它同时解决了一个实际问题——线性空气
// 阻力有闭式的正向解，只有反解需要迭代，所以拆开之后不必像
// Climber_Vision 的 AirResistTrajectory 那样为了反解隐式方程引入 Ceres。
struct Impact {
  double z{0.0};         // 子弹到达该水平距离时的高度，向上为正，单位 m
  double fly_time{0.0};  // 飞行时间，单位 s
};

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

  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

// 真空抛体。x = v0*cos(t)*t，z = v0*sin(t)*t - g*t²/2。
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

  [[nodiscard]] std::string_view name() const noexcept override { return "vacuum"; }
  [[nodiscard]] double gravity() const noexcept { return gravity_; }

private:
  double gravity_;
};

// 线性空气阻力，只作用在水平分量上：dvx/dt = -k*vx，竖直方向仍为自由落体。
//
//   t(x) = (exp(k*x) - 1) / (k * v0 * cos(pitch))
//   z(t) = v0*sin(pitch)*t - g*t²/2
//
// 这是 talos LinearDragModel 的模型。与 Climber_Vision 在两个方向都加阻力
// 的完整模型相比，它牺牲了竖直方向的阻力项，换来正向闭式解——17mm 弹丸在
// 10 m 内竖直速度远小于水平速度，这一项的影响比水平衰减小一个量级。
class LinearDragModel final : public IBallisticModel {
public:
  LinearDragModel(double gravity, double drag_coefficient) noexcept
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

    // k 趋于 0 时上式是 0/0，用真空解的极限收尾，保证 k = 0 严格退化。
    const double horizontal_speed = v0 * cos_pitch;
    const double fly_time = drag_ < 1e-6
                              ? range / horizontal_speed
                              : std::expm1(drag_ * range) / (drag_ * horizontal_speed);
    if (fly_time < 0.0 || !std::isfinite(fly_time)) {
      return std::nullopt;
    }

    const double z =
      v0 * std::sin(pitch) * fly_time - 0.5 * gravity_ * fly_time * fly_time;
    return Impact{z, fly_time};
  }

  [[nodiscard]] std::string_view name() const noexcept override { return "linear_drag"; }
  [[nodiscard]] double gravity() const noexcept { return gravity_; }
  [[nodiscard]] double drag() const noexcept { return drag_; }

private:
  double gravity_;
  double drag_;
};

}  // namespace L4Planning
