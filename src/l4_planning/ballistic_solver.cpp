#include "l4_planning/ballistic_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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
    model_ = std::make_shared<const LinearDragModel>(
      config_.gravity, config_.drag_coefficient);
  } else {
    model_ = std::make_shared<const VacuumModel>(config_.gravity);
  }
}

std::optional<double> BallisticSolver::vacuumPitch(
  double d, double h, double bullet_speed) const
{
  // 真空抛体：把 tan(pitch) 当未知量得到一元二次方程
  //   a * tan²θ + b * tanθ + c = 0
  //   a = g*d²/(2*v0²)，b = -d，c = a + h
  const double a = config_.gravity * d * d / (2.0 * bullet_speed * bullet_speed);
  if (a < 1e-12) {
    return std::nullopt;
  }

  const double b = -d;
  const double c = a + h;
  const double discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    // 判别式为负表示该初速打不到这个距离和高度，属于正常的物理无解。
    return std::nullopt;
  }

  const double sqrt_discriminant = std::sqrt(discriminant);
  const double pitch_high = std::atan((-b + sqrt_discriminant) / (2.0 * a));
  const double pitch_low = std::atan((-b - sqrt_discriminant) / (2.0 * a));

  // 两个解分别对应高抛和低抛。取飞行时间短的低弧：目标外推误差随飞行时间
  // 线性放大，而且高抛更容易撞到场地限高。
  const auto flight_time = [d, bullet_speed](double pitch) {
    const double horizontal_speed = bullet_speed * std::cos(pitch);
    return horizontal_speed > 1e-6 ? d / horizontal_speed
                                   : std::numeric_limits<double>::infinity();
  };
  const double pitch =
    flight_time(pitch_low) <= flight_time(pitch_high) ? pitch_low : pitch_high;
  return std::isfinite(pitch) ? std::optional<double>{pitch} : std::nullopt;
}

Ballistic BallisticSolver::solve(double d, double h, double bullet_speed) const
{
  Ballistic result;

  if (!std::isfinite(d) || !std::isfinite(h) || !std::isfinite(bullet_speed) ||
      d < kMinDistance || bullet_speed < kMinSolvableSpeed) {
    return result;
  }

  const auto guess = vacuumPitch(d, h, bullet_speed);
  if (!guess.has_value()) {
    return result;
  }

  const auto finish = [&result, this](double pitch, const Impact& impact) {
    if (std::abs(pitch) > config_.max_pitch || !std::isfinite(pitch) ||
        !std::isfinite(impact.fly_time) || impact.fly_time <= 0.0) {
      return;
    }
    // yaw 与弹道无关，由调用方从水平分量直接求得，这里保持 0。
    result.pitch = pitch;
    result.fly_time = impact.fly_time;
    result.valid = true;
  };

  const auto guess_impact = model_->impact(d, *guess, bullet_speed);
  if (!guess_impact.has_value()) {
    return result;
  }

  // 真空模型的闭式解就是精确解，落点高度已经等于 h，无需迭代。
  if (config_.drag_coefficient <= 1e-6) {
    finish(*guess, *guess_impact);
    return result;
  }

  // 高度补偿迭代：抬高"名义目标高度"直到实际落点回到真实高度。
  double aim_height = h;
  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const auto pitch = vacuumPitch(d, aim_height, bullet_speed);
    if (!pitch.has_value() || std::abs(*pitch) > config_.max_pitch) {
      return result;
    }

    const auto impact = model_->impact(d, *pitch, bullet_speed);
    if (!impact.has_value()) {
      return result;
    }

    const double error = h - impact->z;
    if (std::abs(error) < config_.height_tolerance) {
      finish(*pitch, *impact);
      return result;
    }
    aim_height += error;
  }

  // 未收敛说明该初速在这个距离上打不到目标高度，不输出可疑解。
  return result;
}

double BallisticSolver::solvePitch(double distance, double height, double bullet_speed) const
{
  const Ballistic result = solve(distance, height, bullet_speed);
  return result.valid ? result.pitch : 0.0;
}

}  // namespace L4Planning
