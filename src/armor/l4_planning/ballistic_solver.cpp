#include "l4_planning/ballistic_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace L4Planning {

namespace {

constexpr double kAngleBoundaryMargin = 1e-9; // 避开阻力模型的无穷飞行时间边界
constexpr int kGoldenSectionIterations = 80; // 搜索高度残差最高点的固定次数
constexpr int kBisectionIterations = 80; // 求低/高弹道根的固定次数

// 两种内部弹道模型统一返回的最小结果。
struct Trajectory {
  double pitch{0.0};    // 最终选中的弹道俯仰角，rad
  double fly_time{0.0}; // 弹丸飞行时间，s
  bool valid{false};    // 目标是否可达且数值结果有效
};

// 求解无空气阻力条件下的弹道。优先返回低弹道；低弹道数值校验失败时，
// 再尝试高弹道。低弹道根使用有理式，避免 v^2 与 sqrt(discriminant)
// 很接近时发生严重的浮点消减。
[[nodiscard]] Trajectory solveVacuumTrajectory(
  double distance,
  double height,
  double bullet_speed,
  double gravity)
{
  Trajectory trajectory;
  if (!std::isfinite(distance) || !std::isfinite(height)
      || !std::isfinite(bullet_speed) || !std::isfinite(gravity)
      || distance <= 0.0
      || bullet_speed <= 0.0 || gravity <= 0.0) {
    return trajectory;
  }

  const double speed_squared = bullet_speed * bullet_speed;
  const double speed_fourth = speed_squared * speed_squared;
  const double gravity_term =
    gravity * (gravity * distance * distance + 2.0 * height * speed_squared);
  double discriminant = speed_fourth - gravity_term;
  if (!std::isfinite(discriminant)) {
    return trajectory;
  }

  // 理论上恰好为零的判别式可能因舍入误差得到一个很小的负数。
  const double scale =
    std::max({1.0, std::abs(speed_fourth), std::abs(gravity_term)});
  const double discriminant_tolerance =
    32.0 * std::numeric_limits<double>::epsilon() * scale;
  if (discriminant < -discriminant_tolerance) {
    return trajectory;
  }
  discriminant = std::max(0.0, discriminant);

  const double root = std::sqrt(discriminant);
  const auto make_trajectory = [=](double tan_pitch) {
    Trajectory candidate;
    if (!std::isfinite(tan_pitch)) {
      return candidate;
    }

    const double pitch = std::atan(tan_pitch);
    const double horizontal_speed = bullet_speed * std::cos(pitch);
    if (!std::isfinite(pitch) || !std::isfinite(horizontal_speed)
        || horizontal_speed <= 0.0) {
      return candidate;
    }

    const double fly_time = distance / horizontal_speed;
    if (!std::isfinite(fly_time) || fly_time <= 0.0) {
      return candidate;
    }

    candidate.pitch = pitch;
    candidate.fly_time = fly_time;
    candidate.valid = true;
    return candidate;
  };

  // 先尝试低弹道根。这里使用有理化后的等价公式降低浮点消减。
  const double low_denominator = distance * (speed_squared + root);
  if (std::isfinite(low_denominator) && low_denominator > 0.0) {
    const double low_tan_pitch =
      (gravity * distance * distance + 2.0 * height * speed_squared)
      / low_denominator;
    const auto low_trajectory = make_trajectory(low_tan_pitch);
    if (low_trajectory.valid) {
      return low_trajectory;
    }
  }

  // 低弹道未通过数值校验时，使用另一个解析根尝试高弹道。
  const double high_denominator = gravity * distance;
  if (!std::isfinite(high_denominator) || high_denominator <= 0.0) {
    return trajectory;
  }
  const double high_tan_pitch = (speed_squared + root) / high_denominator;
  return make_trajectory(high_tan_pitch);
}

// 线性阻力模型：
//   dv/dt = -k*v + [0, 0, -g]
//
// 令 A(t)=(1-exp(-k*t))/k，则二维弹道满足：
//   distance = v0*cos(pitch)*A(t)
//   height   = v0*sin(pitch)*A(t) - g/k*(t-A(t))
//
// 对给定 pitch，可由第一式解析得到 t，再对第二式的高度残差求根。
// 高度残差在允许角度区间内先上升后下降；先用黄金分割找到最高点，
// 再在左半区二分求低弹道。低弹道校验失败时，在右半区求高弹道。
[[nodiscard]] Trajectory solveLinearDragTrajectory(
  double distance,
  double height,
  double bullet_speed,
  double gravity,
  double drag_coefficient)
{
  Trajectory trajectory;
  if (!std::isfinite(distance) || !std::isfinite(height)
      || !std::isfinite(bullet_speed) || !std::isfinite(gravity)
      || !std::isfinite(drag_coefficient)
      || distance <= 0.0
      || bullet_speed <= 0.0 || gravity <= 0.0
      || drag_coefficient <= 0.0) {
    return trajectory;
  }

  const double horizontal_ratio =
    drag_coefficient * distance / bullet_speed;
  if (!std::isfinite(horizontal_ratio)
      || horizontal_ratio <= 0.0
      || horizontal_ratio >= 1.0) {
    // 线性阻力下水平位移上限为 v0/k；超过该距离必然不可达。
    return trajectory;
  }

  const double maximum_pitch = std::acos(horizontal_ratio);
  if (!std::isfinite(maximum_pitch)
      || maximum_pitch <= kAngleBoundaryMargin) {
    return trajectory;
  }

  const double lower_pitch = -maximum_pitch + kAngleBoundaryMargin;
  const double upper_pitch = maximum_pitch - kAngleBoundaryMargin;
  const auto evaluate = [=](double pitch, double* fly_time = nullptr) {
    const double cosine = std::cos(pitch);
    if (!std::isfinite(cosine) || cosine <= 0.0) {
      return -std::numeric_limits<double>::infinity();
    }

    const double ratio = horizontal_ratio / cosine;
    if (!std::isfinite(ratio) || ratio <= 0.0 || ratio >= 1.0) {
      return -std::numeric_limits<double>::infinity();
    }

    const double time = -std::log1p(-ratio) / drag_coefficient;
    const double attenuation_time =
      -std::expm1(-drag_coefficient * time) / drag_coefficient;
    const double drag_argument = drag_coefficient * time;
    // time-A(t) 等价于 (x+exp(-x)-1)/k。x 很小时直接相减会丢失
    // 有效数字，因此使用泰勒展开保证 k->0 时连续收敛到真空模型。
    const double gravity_drop_numerator =
      std::abs(drag_argument) < 1e-3
        ? drag_argument * drag_argument
            * (0.5 + drag_argument
                * (-1.0 / 6.0 + drag_argument
                    * (1.0 / 24.0 - drag_argument / 120.0)))
        : drag_argument + std::expm1(-drag_argument);
    const double gravity_drop =
      gravity * gravity_drop_numerator
      / (drag_coefficient * drag_coefficient);
    const double predicted_height =
      bullet_speed * std::sin(pitch) * attenuation_time
      - gravity_drop;
    if (!std::isfinite(time) || time <= 0.0
        || !std::isfinite(predicted_height)) {
      return -std::numeric_limits<double>::infinity();
    }

    if (fly_time != nullptr) {
      *fly_time = time;
    }
    return predicted_height - height;
  };

  // 寻找高度残差最大的位置，用它判断是否可达并分隔低/高弹道。
  constexpr double golden_ratio = 0.6180339887498948482;
  double search_left = lower_pitch;
  double search_right = upper_pitch;
  double left_probe =
    search_right - golden_ratio * (search_right - search_left);
  double right_probe =
    search_left + golden_ratio * (search_right - search_left);
  double left_value = evaluate(left_probe);
  double right_value = evaluate(right_probe);
  for (int i = 0; i < kGoldenSectionIterations; ++i) {
    if (left_value < right_value) {
      search_left = left_probe;
      left_probe = right_probe;
      left_value = right_value;
      right_probe =
        search_left + golden_ratio * (search_right - search_left);
      right_value = evaluate(right_probe);
    } else {
      search_right = right_probe;
      right_probe = left_probe;
      right_value = left_value;
      left_probe =
        search_right - golden_ratio * (search_right - search_left);
      left_value = evaluate(left_probe);
    }
  }

  const double peak_pitch = 0.5 * (search_left + search_right);
  const double peak_residual = evaluate(peak_pitch);
  const double residual_tolerance =
    1e-9 * std::max({1.0, distance, std::abs(height)});
  if (!std::isfinite(peak_residual)
      || peak_residual < -residual_tolerance) {
    return trajectory;
  }

  const auto make_trajectory = [&](double pitch) {
    Trajectory candidate;
    double fly_time = 0.0;
    const double final_residual = evaluate(pitch, &fly_time);
    if (!std::isfinite(pitch) || !std::isfinite(fly_time)
        || fly_time <= 0.0 || !std::isfinite(final_residual)
        || std::abs(final_residual) > 10.0 * residual_tolerance) {
      return candidate;
    }

    candidate.pitch = pitch;
    candidate.fly_time = fly_time;
    candidate.valid = true;
    return candidate;
  };

  if (peak_residual <= residual_tolerance) {
    // 与最大点相切时低、高弹道重合。
    return make_trajectory(peak_pitch);
  }

  // 左侧边界的弹道时间趋于无穷且落点趋于负无穷，因此它与残差
  // 最大点之间包围的第一个根就是低弹道根。
  {
    double root_left = lower_pitch;
    double root_right = peak_pitch;
    for (int i = 0; i < kBisectionIterations; ++i) {
      const double middle = 0.5 * (root_left + root_right);
      if (evaluate(middle) < 0.0) {
        root_left = middle;
      } else {
        root_right = middle;
      }
    }
    const auto low_trajectory =
      make_trajectory(0.5 * (root_left + root_right));
    if (low_trajectory.valid) {
      return low_trajectory;
    }
  }

  // 低弹道未通过最终校验时，在峰值右侧求第二个根作为高弹道回退。
  double root_left = peak_pitch;
  double root_right = upper_pitch;
  for (int i = 0; i < kBisectionIterations; ++i) {
    const double middle = 0.5 * (root_left + root_right);
    if (evaluate(middle) >= 0.0) {
      root_left = middle;
    } else {
      root_right = middle;
    }
  }
  return make_trajectory(0.5 * (root_left + root_right));
}

}  // namespace

BallisticSolution BallisticSolver::solve(const BallisticRequest& request) const
{
  BallisticSolution solution;
  if (!request.target_position_barrel.allFinite()) {
    return solution;
  }

  const double x = request.target_position_barrel.x();
  const double y = request.target_position_barrel.y();
  const double z = request.target_position_barrel.z();
  const double horizontal_distance = std::hypot(x, y);
  const auto trajectory = request.enable_air_resistance
    ? solveLinearDragTrajectory(
        horizontal_distance,
        z,
        request.bullet_speed,
        request.gravity,
        request.linear_drag_coefficient)
    : solveVacuumTrajectory(
      horizontal_distance, z, request.bullet_speed, request.gravity);
  if (!trajectory.valid) {
    return solution;
  }

  solution.yaw = std::atan2(y, x);
  solution.pitch = trajectory.pitch;
  solution.fly_time = trajectory.fly_time;
  solution.valid = std::isfinite(solution.yaw);
  solution.used_air_resistance = request.enable_air_resistance;
  return solution;
}

}  // namespace L4Planning
