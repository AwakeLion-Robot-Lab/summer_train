#include "l4_planning/ballistic.hpp"

#include <cmath>
#include <limits>

namespace L4Planning {
namespace {

std::optional<Launch> solveByEffectiveRange(
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

  const double root = std::sqrt(discriminant);
  const double pitch_low = std::atan((a - root) / (2.0 * b));
  const double pitch_high = std::atan((a + root) / (2.0 * b));
  const auto flightTime = [a, v0](double pitch) {
    const double horizontal_speed = v0 * std::cos(pitch);
    return horizontal_speed > 1e-6
             ? a / horizontal_speed
             : std::numeric_limits<double>::infinity();
  };

  const double time_low = flightTime(pitch_low);
  const double time_high = flightTime(pitch_high);
  // 两条弹道都可达时选择飞行时间更短的一条，减少预测误差累积。
  const bool use_low = time_low <= time_high;
  const Launch result{
    use_low ? pitch_low : pitch_high,
    use_low ? time_low : time_high};
  if (!std::isfinite(result.pitch) || !std::isfinite(result.fly_time) ||
      result.fly_time <= 0.0) {
    return std::nullopt;
  }
  return result;
}

}  // namespace

std::optional<Launch> IBallisticModel::launch(
  double range, double height, double v0) const noexcept
{
  (void)range;
  (void)height;
  (void)v0;
  return std::nullopt;
}

VacuumModel::VacuumModel(double gravity) noexcept
: gravity_(gravity)
{
}

std::optional<Impact> VacuumModel::impact(
  double range, double pitch, double v0) const noexcept
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

std::optional<Launch> VacuumModel::launch(
  double range, double height, double v0) const noexcept
{
  return solveByEffectiveRange(range, height, v0, gravity_);
}

std::string_view VacuumModel::name() const noexcept
{
  return "vacuum";
}

QuadraticDragModel::QuadraticDragModel(
  double gravity, double drag_coefficient) noexcept
: gravity_(gravity), drag_(drag_coefficient)
{
}

std::optional<Impact> QuadraticDragModel::impact(
  double range, double pitch, double v0) const noexcept
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

std::optional<Launch> QuadraticDragModel::launch(
  double range, double height, double v0) const noexcept
{
  return solveByEffectiveRange(effectiveRange(range), height, v0, gravity_);
}

std::string_view QuadraticDragModel::name() const noexcept
{
  return "quadratic_drag";
}

double QuadraticDragModel::effectiveRange(double range) const noexcept
{
  // 把水平阻力造成的速度衰减折算成等效无阻力距离。
  return drag_ < 1e-6 ? range : std::expm1(drag_ * range) / drag_;
}

}  // namespace L4Planning
