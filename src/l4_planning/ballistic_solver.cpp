#include "l4_planning/ballistic_solver.hpp"

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

  // 高度补偿迭代：抬高"名义目标高度"直到实际落点回到真实高度。内层映射用真空
  // 闭式解，而不是 talos / FYT 的视线角 atan2 —— 视线角完全不含重力，整个下坠
  // 量都要靠迭代累加出来，收敛慢一截。
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

  // 模型给得出闭式反解就直接用。两个内置模型都走这条路：一次算完，精确到机器
  // 精度，没有容差残留，也没有"迭代次数不够就报无解"这个失效模式。
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

double BallisticSolver::solvePitch(double distance, double height, double bullet_speed) const
{
  const Ballistic result = solve(distance, height, bullet_speed);
  return result.valid ? result.pitch : 0.0;
}

}  // namespace L4Planning
