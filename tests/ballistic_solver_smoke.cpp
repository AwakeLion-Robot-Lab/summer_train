#include "l3_planning/ballistic_solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <string_view>

using namespace L4Planning;
namespace {
int failures = 0;
void expect(bool value, std::string_view message)
{
  if (!value) { if (failures < 15) std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

// 独立 long double oracle：全区间黄金分割 + 二分，不使用生产代码的峰值公式或导数。
BallisticSolution referenceDrag(const BallisticRequest& request)
{
  BallisticSolution result;
  const long double r = std::hypot(request.target_position_barrel.x(), request.target_position_barrel.y());
  const long double h = request.target_position_barrel.z(), v = request.bullet_speed;
  const long double k = request.linear_drag_coefficient, g = request.gravity;
  const long double b = k * r / v;
  if (r <= 0 || v <= 0 || g <= 0 || k <= 0 || b <= 0 || b >= 1) return result;
  const long double bound = std::acos(b) - 1e-9L;
  if (bound <= 0) return result;
  const auto residual = [&](long double theta) {
    const long double q = b / std::cos(theta);
    if (q <= 0 || q >= 1) return -std::numeric_limits<long double>::infinity();
    long double drop = -std::log1p(-q) - q;
    if (q < 0.01L) {
      drop = 0;
      long double power = q * q;
      for (int n = 2; n < 14; ++n) { drop += power / n; power *= q; }
    }
    return r * std::tan(theta) - g * drop / (k * k) - h;
  };
  long double lo = -bound, hi = bound;
  constexpr long double golden = 0.618033988749894848204586834365638L;
  long double left = hi - golden * (hi - lo), right = lo + golden * (hi - lo);
  long double fl = residual(left), fr = residual(right);
  for (int n = 0; n < 120; ++n) {
    if (fl < fr) {
      lo = left; left = right; fl = fr;
      right = lo + golden * (hi - lo); fr = residual(right);
    } else {
      hi = right; right = left; fr = fl;
      left = hi - golden * (hi - lo); fl = residual(left);
    }
  }
  const long double peak = (lo + hi) / 2;
  const long double tolerance = 1e-9L * std::max({1.0L, r, std::abs(h)});
  if (residual(peak) < -tolerance) return result;
  long double theta = peak;
  if (residual(peak) > tolerance) {
    lo = -bound; hi = peak;
    for (int n = 0; n < 120; ++n) {
      const long double mid = (lo + hi) / 2;
      if (residual(mid) < 0) lo = mid;
      else hi = mid;
    }
    theta = (lo + hi) / 2;
  }
  result.valid = true;
  result.pitch = theta;
  result.fly_time = -std::log1p(-b / std::cos(theta)) / k;
  return result;
}

void accuracyChecks()
{
  std::mt19937 random(20261109);
  std::uniform_real_distribution<double> range(0.1, 45.0), height(-8.0, 12.0);
  std::uniform_real_distribution<double> speed(12.0, 40.0);
  const BallisticSolver solver;
  double max_angle = 0.0, max_time = 0.0;
  int reachable = 0, fallback = 0;
  for (int i = 0; i < 2400; ++i) {
    BallisticRequest request;
    request.target_position_barrel = {range(random), 0.0, height(random)};
    request.bullet_speed = speed(random);
    request.enable_air_resistance = true;
    const double drag[] = {1e-8, 1e-5, 0.01, 0.5, 1.0, 2.0};
    request.linear_drag_coefficient = drag[i % 6];
    const auto reference = referenceDrag(request);
    const auto cold = solver.solve(request);
    expect(cold.valid == reference.valid, "fast solver agrees with independent reachability oracle");
    if (!cold.valid || !reference.valid) continue;
    ++reachable; fallback += cold.used_fallback;
    max_angle = std::max(max_angle, std::abs(cold.pitch - reference.pitch));
    max_time = std::max(max_time, std::abs(cold.fly_time - reference.fly_time));
    expect(std::abs(cold.pitch - reference.pitch) < 1e-8, "same low ballistic root as independent oracle");
    expect(std::abs(cold.fly_time - reference.fly_time) < 1e-7, "flight time retains numerical accuracy");
    // 不相关、非有限及接近高弹道的初值都不能改变根分支或有效性。
    const double hints[] = {cold.pitch + 0.03, -1.4, 1.4, std::numeric_limits<double>::quiet_NaN()};
    request.initial_pitch = hints[i % 4];
    const auto seeded = solver.solve(request);
    expect(seeded.valid && std::abs(seeded.pitch - reference.pitch) < 1e-8
      && std::abs(seeded.fly_time - reference.fly_time) < 1e-7, "hint changes convergence but not selected root");
  }
  std::cout << "ballistic oracle: reachable=" << reachable << " fallback=" << fallback
    << " max_pitch_rad=" << max_angle << " max_flight_s=" << max_time << '\n';
}

void derivativeChecks()
{
  const BallisticSolver solver;
  double maximum_error = 0.0;
  for (double drag : {0.0, 1e-8, 0.5, 1.5}) {
    for (const Eigen::Vector3d point : {Eigen::Vector3d{0.3, 0.2, -0.5},
        Eigen::Vector3d{4.0, -0.1, 0.2}, Eigen::Vector3d{-6.0, 0.0, 2.0}}) {
      BallisticRequest request;
      request.target_position_barrel = point; request.bullet_speed = 25.0;
      request.enable_air_resistance = drag > 0.0; request.linear_drag_coefficient = drag;
      const auto solution = solver.solve(request);
      const auto derivative = solver.positionJacobian(request, solution);
      expect(solution.valid && derivative.valid, "analytic derivative available in normal flight envelope");
      for (int axis = 0; axis < 3; ++axis) {
        constexpr double step = 1e-4;
        request.target_position_barrel = point;
        request.target_position_barrel[axis] += step;
        const auto plus = solver.solve(request);
        request.target_position_barrel[axis] -= 2 * step;
        const auto minus = solver.solve(request);
        const Eigen::Vector2d numeric{std::remainder(plus.yaw - minus.yaw, 2 * std::numbers::pi) / (2 * step),
          (plus.pitch - minus.pitch) / (2 * step)};
        const double error = (numeric - derivative.angles_position.col(axis)).norm();
        maximum_error = std::max(maximum_error, error);
        expect(plus.valid && minus.valid && error < 2e-6 * (1 + numeric.norm()),
          "analytic xyz derivative agrees with finite difference, including yaw wrap");
      }
    }
  }
  BallisticRequest tangent;
  tangent.bullet_speed = 25.0;
  tangent.target_position_barrel.x() = 625.0 / tangent.gravity;
  const auto solution = solver.solve(tangent);
  expect(solution.valid && !solver.positionJacobian(tangent, solution).valid,
    "singular tangent derivative is rejected");
  std::cout << "analytic derivative: maximum_fd_error=" << maximum_error << '\n';
}

void boundaryChecks()
{
  BallisticRequest request;
  request.target_position_barrel = {4.0, 0.0, 0.0}; request.bullet_speed = 25.0;
  request.enable_air_resistance = true; request.linear_drag_coefficient = 0.5;
  const double r = 4.0, v = 25.0, k = 0.5, g = request.gravity;
  const double peak = std::acos((k * r / v) / std::hypot(1.0, g * r / (v * v))) - std::atan(g * r / (v * v));
  const double t = -std::log1p(-k * r / (v * std::cos(peak))) / k;
  const double a = -std::expm1(-k * t) / k;
  const double maximum_height = v * std::sin(peak) * a - g * (t - a) / k;
  request.target_position_barrel.z() = maximum_height;
  const auto tangent = BallisticSolver{}.solve(request);
  expect(tangent.valid && tangent.used_fallback, "nearly double root uses original solver");
  request.target_position_barrel.z() += 1e-3;
  expect(!BallisticSolver{}.solve(request).valid, "above peak is unreachable");
  request.target_position_barrel.x() = v / k;
  expect(!BallisticSolver{}.solve(request).valid, "horizontal drag distance limit remains rejected");
  request.target_position_barrel.x() = std::numeric_limits<double>::quiet_NaN();
  expect(!BallisticSolver{}.solve(request).valid, "nonfinite target remains rejected");
}
}  // namespace

int main()
{
  accuracyChecks(); derivativeChecks(); boundaryChecks();
  if (!failures) std::cout << "ballistic_solver_smoke: all checks passed\n";
  return failures ? 1 : 0;
}
