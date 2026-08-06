#include "l4_planning/ballistic_solver.hpp"

#include <algorithm>
#include <cmath>

namespace L4Planning {
namespace {

// 取上海地区重力加速度，与参考实现保持一致，便于对拍。
constexpr double kGravity = 9.7833;
// 低于该初速认为裁判系统数据无效；
constexpr double kMinBulletSpeed = 21.0;
// 水平距离过小时俯仰角趋于奇异，直接判无解。
constexpr double kMinDistance = 0.1;

}  // namespace

BallisticSolver::BallisticSolver(double drag_coefficient)
: drag_coefficient_(drag_coefficient)
{
}

Ballistic BallisticSolver::solve(double d, double h, double bullet_speed) const
{
  Ballistic result;

  if (!std::isfinite(d) || !std::isfinite(h) || !std::isfinite(bullet_speed) ||
      d < kMinDistance || bullet_speed < kMinBulletSpeed) {
    return result;
  }

  // 真空抛体：把 tan(pitch) 当未知量得到一元二次方程
  //   a * tan²θ + b * tanθ + c = 0
  //   a = g*d²/(2*v0²)，b = -d，c = a + h
  const double a = kGravity * d * d / (2.0 * bullet_speed * bullet_speed);
  const double b = -d;
  const double c = a + h;

  // a 与 v0² 成反比，正常参数下不会退化；仍然守一次除零。
  if (a < 1e-12) {
    return result;
  }

  const double discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    // 判别式为负表示该初速打不到这个距离和高度，属于正常的物理无解。
    return result;
  }

  const double sqrt_discriminant = std::sqrt(discriminant);
  const double pitch_high = std::atan((-b + sqrt_discriminant) / (2.0 * a));
  const double pitch_low = std::atan((-b - sqrt_discriminant) / (2.0 * a));

  const auto flight_time = [d, bullet_speed](double pitch) {
    const double horizontal_speed = bullet_speed * std::cos(pitch);
    return horizontal_speed > 1e-6 ? d / horizontal_speed
                                   : std::numeric_limits<double>::infinity();
  };

  const double time_high = flight_time(pitch_high);
  const double time_low = flight_time(pitch_low);

  // 两个解分别对应高抛和低抛。取飞行时间短的低弧：目标外推误差随飞行
  // 时间线性放大，而且高抛更容易撞到场地限高。
  const bool prefer_low = time_low <= time_high;
  result.pitch = prefer_low ? pitch_low : pitch_high;
  result.fly_time = prefer_low ? time_low : time_high;

  if (!std::isfinite(result.pitch) || !std::isfinite(result.fly_time)) {
    return result;
  }

  // yaw 与弹道无关，由调用方从水平分量直接求得，这里保持 0。
  result.valid = true;
  return result;
}

double BallisticSolver::solvePitch(double distance, double height, double bullet_speed) const
{
  const Ballistic result = solve(distance, height, bullet_speed);
  return result.valid ? result.pitch : 0.0;
}

}  // namespace L4Planning
