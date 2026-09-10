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

  // 基准取"新板轨迹在本帧的状态"，不是它在 duration 之后的状态。多项式拟的
  // 是**偏差**，基准由调用方每帧重新提供，所以这里只需要偏差的初值。
  const AimState base = after(0.0);
  if (!allFinite(start) || !allFinite(base)) {
    return solution;
  }

  // 偏差的起点 = 旧板轨迹 - 新板轨迹，终点恒为零。位置差先归一化：直接相减
  // 会在目标扫过 ±pi 时解出绕整整一圈的过渡段，云台朝反方向甩。
  const AxisState yaw_start{
    wrapToPi(start.yaw.position - base.yaw.position),
    start.yaw.velocity - base.yaw.velocity,
    start.yaw.acceleration - base.yaw.acceleration};
  const AxisState pitch_start{
    wrapToPi(start.pitch.position - base.pitch.position),
    start.pitch.velocity - base.pitch.velocity,
    start.pitch.acceleration - base.pitch.acceleration};

  solution.duration = duration;
  solution.base = base;
  solution.yaw = Quintic::fit(yaw_start, AxisState{}, duration);
  solution.pitch = Quintic::fit(pitch_start, AxisState{}, duration);
  // 下发角 = 基准 + 偏差，加速度也是两者之和。偏差多项式的峰值不是全部，
  // 还要加上新板轨迹自身的加速度——按三角不等式取上界，宁可把时长解长一点，
  // 也不能谎报"满足加速度限"。
  solution.peak_yaw_acceleration =
    solution.yaw.peakAbsAcceleration() + std::abs(base.yaw.acceleration);
  solution.peak_pitch_acceleration =
    solution.pitch.peakAbsAcceleration() + std::abs(base.pitch.acceleration);
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
  held_for_ = 0.0;
  solution_ = BlendSolution{};
}

// 过渡进行中的基准轨迹：优先用本帧最新的采样器，拿不到才退回提交时刻那份
// 状态做匀加速外推。拿不到只发生在规划本身失败、只靠 blendOnlyPlan 续着走的
// 帧上；过渡最长 200 ms，这段外推的误差远小于一次切板的阶跃。
AimState AimSmoother::liveBase(
  const std::optional<Forecast>& forecast, double tau)
{
  if (forecast && forecast->after) {
    const AimState sampled = forecast->after(0.0);
    if (allFinite(sampled)) {
      last_base_ = sampled;
      last_base_tau_ = tau;
      return sampled;
    }
  }
  // 拿不到本帧采样器时从**上一次拿到的**基准匀加速外推，而不是从提交时刻的
  // 那份。从提交时刻推会在退化的那一帧甩出一个与已推进时间成正比的阶跃，
  // 恰恰是这次重写要消掉的东西。
  const double dt = tau - last_base_tau_;
  const auto extrapolate = [dt](const AxisState& state) {
    return AxisState{
      state.position + state.velocity * dt + 0.5 * state.acceleration * dt * dt,
      state.velocity + state.acceleration * dt,
      state.acceleration};
  };
  return AimState{extrapolate(last_base_.yaw), extrapolate(last_base_.pitch)};
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
      // 过渡进行中：**系数不重拟**。每帧按剩余时间重拟的话，剩余时间趋零时
      // 加速度按 1/T^2 发散；确定性正是显式搜索相对 MPC 的优势。
      //
      // 但基准轨迹必须取本帧最新的。多项式表示的是"相对新板轨迹的偏差"，
      // 偏差按固定系数衰减到零，基准跟着 EKF 每帧更新——于是过渡结束时输出
      // 恒等于新板轨迹本身，结构上不可能再有收尾阶跃。
      //
      // 早先的写法把终点 after(duration) 在提交那一刻就冻结，多项式对着一个
      // 越来越旧的预测走完全程，收尾交回真实轨迹时必然甩一下：sp demo 回放上
      // 实测收尾阶跃中位 1.29 度、最大 7.44 度，整段命令跳变反而比不做过渡还差。
      const double clamped = std::max(0.0, tau);
      const AimState base = liveBase(forecast, clamped);
      output.yaw = wrapToPi(base.yaw.position + solution_.yaw.position(clamped));
      output.pitch =
        wrapToPi(base.pitch.position + solution_.pitch.position(clamped));
      fillStatus(output);
      return output;
    }
    // 时长走完了，但射击轨迹未必已经切到目标板上——切板时刻是预测出来的，
    // 预测偏晚就会出现"过渡先跑完、切板还没发生"。这时交还控制权等于把云台
    // 从已经奔到的新板拽回旧板，一个阶跃变成两个。停在基准轨迹上继续等。
    //
    // 等待有上限：预测彻底落空时不能永远不交还，多等一个 max_duration 就走。
    held_for_ = tau - solution_.duration;
    const bool arrived = forecast && forecast->destination_selected;
    if (!arrived && held_for_ < limits_.max_duration) {
      const AimState base = liveBase(forecast, tau);
      output.yaw = wrapToPi(base.yaw.position);
      output.pitch = wrapToPi(base.pitch.position);
      fillStatus(output);
      return output;
    }
    // 射击轨迹已经是目标板，或者等够了。交还控制权，本帧起可以提交下一段。
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
  last_base_ = committed.base;
  last_base_tau_ = 0.0;
  start_time_ = now;
  // 过渡终点晚于切板时刻的量。晚一点是帧量化的必然：切板时刻按前视网格离散、
  // 帧周期本身还在抖，触发条件几乎不可能正好卡在等号上。
  late_by_ = duration - forecast->switch_time;
  held_for_ = 0.0;
  active_ = true;

  // tau = 0 处偏差恰好等于 before - after(0)，加回基准就是 before 本身，
  // 与本帧射击轨迹三阶连续，所以这里直接求值不会产生跳变。
  output.yaw =
    wrapToPi(solution_.base.yaw.position + solution_.yaw.position(0.0));
  output.pitch =
    wrapToPi(solution_.base.pitch.position + solution_.pitch.position(0.0));
  fillStatus(output);
  return output;
}

}  // namespace L4Planning
