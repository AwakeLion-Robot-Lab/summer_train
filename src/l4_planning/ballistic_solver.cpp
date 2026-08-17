#include "l4_planning/ballistic.hpp"

#include <algorithm>
#include <cmath>

namespace L4Planning {
namespace {

// 水平距离过小时俯仰角趋于奇异，直接判无解。
constexpr double kMinDistance = 0.1;
// 只拒绝数学上无解的初速，不承担业务门限。
constexpr double kMinSolvableSpeed = 1e-3;

}  // namespace

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

}  // namespace L4Planning
