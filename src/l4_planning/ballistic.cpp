#include "l4_planning/ballistic.hpp"

#include <algorithm>
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

// 水平距离过小时俯仰角趋于奇异，直接判无解。
constexpr double kMinDistance = 0.1;
// 只拒绝数学上无解的初速，不承担业务门限。
constexpr double kMinSolvableSpeed = 1e-3;

}  // namespace

// ---- 弹道模型：给定发射角求落点，或给定落点反解发射角 ----

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
// ---- 求解器：选定模型，必要时用高度补偿迭代兜底 ----

BallisticSolver::BallisticSolver(BallisticConfig config)
: config_(config)
{
  if (config_.drag_coefficient > 1e-6) {
    model_ = std::make_shared<const QuadraticDragModel>(
      config_.gravity, config_.drag_coefficient);
  } else {
    model_ = std::make_shared<const VacuumModel>(config_.gravity);
  }
}

Ballistic BallisticSolver::solve(double d, double h, double bullet_speed) const
{
  Ballistic result;

  if (!std::isfinite(d) || !std::isfinite(h) || !std::isfinite(bullet_speed) ||
      d < kMinDistance || bullet_speed < kMinSolvableSpeed) {
    return result;
  }

  // 模型能直接反解时优先使用闭式结果；仅在模型不提供反解时才做高度补偿迭代。
  if (const auto launch = model_->launch(d, h, bullet_speed); launch.has_value()) {
    if (std::abs(launch->pitch) > config_.max_pitch ||
        !std::isfinite(launch->pitch) || !std::isfinite(launch->fly_time) ||
        launch->fly_time <= 0.0) {
      return result;
    }
    // yaw 与弹道无关，由调用方从水平分量直接求得，这里保持 0。
    result.pitch = launch->pitch;
    result.fly_time = launch->fly_time;
    result.valid = true;
    return result;
  }

  return solveByHeightCompensation(d, h, bullet_speed);
}

Ballistic BallisticSolver::solveByHeightCompensation(
  double d, double h, double bullet_speed) const
{
  Ballistic result;

  // 高度补偿迭代：抬高名义目标高度，直到所选模型的实际落点回到真实高度。
  // 真空闭式解只负责给出每一轮的发射角初值。
  const VacuumModel seed(config_.gravity);
  double aim_height = h;

  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const auto guess = seed.launch(d, aim_height, bullet_speed);
    if (!guess.has_value() || std::abs(guess->pitch) > config_.max_pitch) {
      return result;
    }

    const auto impact = model_->impact(d, guess->pitch, bullet_speed);
    if (!impact.has_value()) {
      return result;
    }

    const double error = h - impact->z;
    if (std::abs(error) < config_.height_tolerance) {
      if (!std::isfinite(impact->fly_time) || impact->fly_time <= 0.0) {
        return result;
      }
      result.pitch = guess->pitch;
      result.fly_time = impact->fly_time;
      result.valid = true;
      return result;
    }
    aim_height += error;
  }

  // 未收敛说明该初速在这个距离上打不到目标高度，不输出可疑解。
  return result;
}

}  // namespace L4Planning
