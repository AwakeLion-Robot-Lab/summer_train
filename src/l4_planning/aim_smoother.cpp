#include "l4_planning/aim_smoother.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace L4Planning {
namespace {

// 与 L6Telemetry::limit_rad 同语义。这里重写一份是为了让这个编译单元不牵扯
// Eigen 和 OpenCV——math.hpp 两个都要，而本类的全部数值都是标量，脱开之后
// 才能进 standalone_tests 离线验证。
double wrapToPi(double angle)
{
  if (!std::isfinite(angle)) {
    return angle;
  }
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  double wrapped = std::remainder(angle, kTwoPi);
  if (wrapped <= -std::numbers::pi) {
    wrapped += kTwoPi;
  }
  return wrapped;
}

bool allFinite(const AxisState& state)
{
  return std::isfinite(state.position) && std::isfinite(state.velocity) &&
    std::isfinite(state.acceleration);
}

bool allFinite(const AimState& state)
{
  return allFinite(state.yaw) && allFinite(state.pitch);
}

}  // namespace

Quintic Quintic::fit(
  const AxisState& start, const AxisState& end, double duration)
{
  Quintic quintic;

  // 时长非正时整段塌成起点。duration 一并置 0，让调用方通过 duration 就能
  // 认出这是退化解——否则峰值加速度算出 0，看起来反而"最可行"。
  if (!(duration > 0.0) || !std::isfinite(duration)) {
    quintic.coefficient[0] = start.position;
    return quintic;
  }

  const double t1 = duration;
  const double t2 = t1 * t1;
  const double t3 = t2 * t1;
  const double t4 = t3 * t1;
  const double t5 = t4 * t1;

  // tau = 0 处的三个条件直接给出前三个系数。
  quintic.duration = duration;
  quintic.coefficient[0] = start.position;
  quintic.coefficient[1] = start.velocity;
  quintic.coefficient[2] = 0.5 * start.acceleration;

  // 末端条件减去"按起点二阶外推、什么都不做"的结果，剩下的才是三、四、五
  // 次项要补的量。等价于直接解 3x3，但式子里没有大数相消。
  const double dp =
    end.position - start.position - start.velocity * t1 - 0.5 * start.acceleration * t2;
  const double dv = end.velocity - start.velocity - start.acceleration * t1;
  const double da = end.acceleration - start.acceleration;

  quintic.coefficient[3] = (20.0 * dp - 8.0 * dv * t1 + da * t2) / (2.0 * t3);
  quintic.coefficient[4] = (-30.0 * dp + 14.0 * dv * t1 - 2.0 * da * t2) / (2.0 * t4);
  quintic.coefficient[5] = (12.0 * dp - 6.0 * dv * t1 + da * t2) / (2.0 * t5);

  return quintic;
}

double Quintic::position(double tau) const
{
  const double* c = coefficient;
  return c[0] + tau * (c[1] + tau * (c[2] + tau * (c[3] + tau * (c[4] + tau * c[5]))));
}

double Quintic::velocity(double tau) const
{
  const double* c = coefficient;
  return c[1] +
    tau * (2.0 * c[2] + tau * (3.0 * c[3] + tau * (4.0 * c[4] + tau * 5.0 * c[5])));
}

double Quintic::acceleration(double tau) const
{
  const double* c = coefficient;
  return 2.0 * c[2] + tau * (6.0 * c[3] + tau * (12.0 * c[4] + tau * 20.0 * c[5]));
}

double Quintic::peakAbsAcceleration() const
{
  if (!(duration > 0.0)) {
    return 0.0;
  }

  double peak =
    std::max(std::abs(acceleration(0.0)), std::abs(acceleration(duration)));

  const auto consider = [&](double tau) {
    if (std::isfinite(tau) && tau > 0.0 && tau < duration) {
      peak = std::max(peak, std::abs(acceleration(tau)));
    }
  };

  // a'(tau) = 6c3 + 24c4 tau + 60c5 tau^2，内部极值就是它的实根。
  const double a = 60.0 * coefficient[5];
  const double b = 24.0 * coefficient[4];
  const double c = 6.0 * coefficient[3];

  if (std::abs(a) < 1e-12) {
    if (std::abs(b) > 1e-12) {
      consider(-c / b);
    }
    return peak;
  }

  const double discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    return peak;
  }

  // 用 q = -(b + sign(b)*sqrt(D))/2 取根，避免 b 与 sqrt(D) 接近时
  // (-b + sqrt(D)) 相消掉有效位。
  const double q = -0.5 * (b + std::copysign(std::sqrt(discriminant), b));
  consider(q / a);
  if (std::abs(q) > 1e-300) {
    consider(c / q);
  }
  return peak;
}

BlendSolution fitBlend(
  const AimState& start,
  const TrajectorySampler& after,
  double duration,
  const BlendLimits& limits)
{
  BlendSolution solution;
  if (!after || !(duration > 0.0) || !std::isfinite(duration)) {
    return solution;
  }

  const AimState end = after(duration);
  if (!allFinite(start) || !allFinite(end)) {
    return solution;
  }

  // 位置差先归一化再加回起点，把两端拉到同一支连续角度上。直接把两个绝对
  // 角丢给 fit()，目标扫过 ±pi 时会解出绕整整一圈的过渡段，云台朝反方向甩。
  AxisState yaw_end = end.yaw;
  yaw_end.position =
    start.yaw.position + wrapToPi(end.yaw.position - start.yaw.position);
  AxisState pitch_end = end.pitch;
  pitch_end.position =
    start.pitch.position + wrapToPi(end.pitch.position - start.pitch.position);

  solution.duration = duration;
  solution.yaw = Quintic::fit(start.yaw, yaw_end, duration);
  solution.pitch = Quintic::fit(start.pitch, pitch_end, duration);
  solution.peak_yaw_acceleration = solution.yaw.peakAbsAcceleration();
  solution.peak_pitch_acceleration = solution.pitch.peakAbsAcceleration();
  solution.acceleration_limited =
    solution.peak_yaw_acceleration > limits.max_yaw_acceleration ||
    solution.peak_pitch_acceleration > limits.max_pitch_acceleration;
  solution.valid = true;
  return solution;
}

BlendSolution solveBlend(
  const AimState& start,
  const TrajectorySampler& after,
  const BlendLimits& limits)
{
  const double shortest = std::max(1e-3, limits.min_duration);
  const double longest = std::max(shortest, limits.max_duration);

  // 最短过渡就不超限：没有必要减速，直接用它，重合度损失最小。
  BlendSolution best = fitBlend(start, after, shortest, limits);
  if (!best.valid || !best.acceleration_limited) {
    return best;
  }

  // 拉到上限仍然超限：云台能力真的不够。如实返回，让上层把标志送进遥测，
  // 而不是继续加长过渡去掩盖——加长只会把重合度也一起赔进去。
  BlendSolution feasible = fitBlend(start, after, longest, limits);
  if (!feasible.valid || feasible.acceleration_limited) {
    return feasible.valid ? feasible : best;
  }

  double infeasible_duration = shortest;
  double feasible_duration = longest;
  for (int iteration = 0; iteration < limits.search_iterations; ++iteration) {
    const double middle = 0.5 * (infeasible_duration + feasible_duration);
    const BlendSolution candidate = fitBlend(start, after, middle, limits);
    if (!candidate.valid) {
      break;
    }
    if (candidate.acceleration_limited) {
      infeasible_duration = middle;
    } else {
      feasible_duration = middle;
      feasible = candidate;
    }
  }
  return feasible;
}

AimSmoother::AimSmoother(BlendLimits limits) noexcept
: limits_(limits)
{
}

void AimSmoother::fillStatus(Output& output) const noexcept
{
  output.blending = true;
  output.peak_yaw_acceleration = solution_.peak_yaw_acceleration;
  output.peak_pitch_acceleration = solution_.peak_pitch_acceleration;
  output.acceleration_limited = solution_.acceleration_limited;
  output.late_by = late_by_;
}

void AimSmoother::reset() noexcept
{
  active_ = false;
  late_by_ = 0.0;
  solution_ = BlendSolution{};
}

AimSmoother::Output AimSmoother::update(
  TimePoint now,
  double shoot_yaw,
  double shoot_pitch,
  const std::optional<Forecast>& forecast)
{
  Output output;
  output.yaw = shoot_yaw;
  output.pitch = shoot_pitch;

  if (active_) {
    const double tau = std::chrono::duration<double>(now - start_time_).count();
    if (tau < solution_.duration) {
      // 过渡进行中：只求值，不重新规划。每帧按剩余时间重拟的话，剩余时间
      // 趋零时加速度按 1/T^2 发散；确定性正是显式搜索相对 MPC 的优势。
      const double clamped = std::max(0.0, tau);
      output.yaw = wrapToPi(solution_.yaw.position(clamped));
      output.pitch = wrapToPi(solution_.pitch.position(clamped));
      fillStatus(output);
      return output;
    }
    // 过渡结束，回到射击轨迹。本帧起立刻可以重新提交下一段。
    reset();
  }

  if (!forecast || !std::isfinite(forecast->switch_time)) {
    return output;
  }

  // 切板还远到超过最长过渡时长时，最小可行时长必然更短，不可能满足下面的
  // 提交条件。先挡掉，省去这一帧的二分——那是本类唯一有成本的部分。
  if (forecast->switch_time > limits_.max_duration) {
    return output;
  }

  const BlendSolution minimal =
    solveBlend(forecast->before, forecast->after, limits_);
  if (!minimal.valid) {
    return output;
  }

  // 距切板还够远，先不提交，下一帧再看。触发点就是最小可行时长本身，不留
  // 余量：提早提交只会让时长取到偏大的值，把每一段过渡都拉长。
  if (forecast->switch_time > minimal.duration) {
    return output;
  }

  // 时长恒取最小可行值：过渡越短，偏离射击轨迹的时间越短，重合度越高。
  //
  // 终点因此落在切板时刻**或其之后**，绝不会落在之前。提前结束是不行的：
  // 那时真正的切板还没发生，输出会先从新板轨迹掉回旧板，到切板时再跳一次，
  // 一个阶跃变成两个，比不做过渡还糟。
  const double duration = minimal.duration;
  const BlendSolution committed =
    fitBlend(forecast->before, forecast->after, duration, limits_);
  if (!committed.valid) {
    return output;
  }

  solution_ = committed;
  start_time_ = now;
  // 过渡终点晚于切板时刻的量。晚一点是帧量化的必然：切板时刻按前视网格离散、
  // 帧周期本身还在抖，触发条件几乎不可能正好卡在等号上。
  late_by_ = duration - forecast->switch_time;
  active_ = true;

  // tau = 0 处多项式等于 before(0)，与本帧射击轨迹三阶连续，所以这里直接
  // 求值不会产生跳变。
  output.yaw = wrapToPi(solution_.yaw.position(0.0));
  output.pitch = wrapToPi(solution_.pitch.position(0.0));
  fillStatus(output);
  return output;
}

}  // namespace L4Planning
