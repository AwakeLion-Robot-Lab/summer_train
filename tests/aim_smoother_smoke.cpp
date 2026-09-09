#include "l4_planning/aim_smoother.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <optional>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what)
{
  if (!condition) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
  }
}

void checkNear(double got, double want, double tolerance, const char* what)
{
  if (!(std::abs(got - want) <= tolerance)) {
    std::fprintf(
      stderr, "FAIL: %s (got %.12g, want %.12g, tol %.3g)\n", what, got, want,
      tolerance);
    ++failures;
  }
}

double wrapToPi(double angle)
{
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  double wrapped = std::remainder(angle, kTwoPi);
  if (wrapped <= -std::numbers::pi) {
    wrapped += kTwoPi;
  }
  return wrapped;
}

// 合成的小陀螺射击轨迹：一块装甲板绕车心转，方位角就是从枪口看过去的
// atan2。bias 用来把整条轨迹平移到 ±pi 附近，专门验角度归一化。
struct Plate {
  double distance{3.0};  // 车心距离，m
  double radius{0.25};   // 装甲板半径，m
  double omega{10.0};    // 整车 yaw 角速度，rad/s
  double phase{0.0};     // t = 0 时的板面法向角
  double height{0.0};    // 板心相对枪口的高度，m
  double bias{0.0};      // 方位偏置
};

double plateYaw(const Plate& plate, double t)
{
  const double theta = plate.phase + plate.omega * t;
  const double x = plate.distance + plate.radius * std::cos(theta);
  const double y = plate.radius * std::sin(theta);
  return wrapToPi(std::atan2(y, x) + plate.bias);
}

double platePitch(const Plate& plate, double t)
{
  const double theta = plate.phase + plate.omega * t;
  const double x = plate.distance + plate.radius * std::cos(theta);
  const double y = plate.radius * std::sin(theta);
  return std::atan2(plate.height, std::hypot(x, y));
}

// 位置取归一化值，速度和加速度按**归一化后的差分**算。真实实现里前向采样表
// 也必须这么做：aim yaw 出自 atan2，天然落在 (-pi, pi]，直接差分会在 ±pi
// 处得到一个 2pi/dt 的假尖峰。
L4Planning::AxisState differentiate(
  double (*value)(const Plate&, double), const Plate& plate, double t)
{
  constexpr double kStep = 5e-4;
  const double middle = value(plate, t);
  const double back = value(plate, t - kStep);
  const double front = value(plate, t + kStep);

  L4Planning::AxisState state;
  state.position = middle;
  state.velocity = wrapToPi(front - back) / (2.0 * kStep);
  state.acceleration =
    (wrapToPi(front - middle) - wrapToPi(middle - back)) / (kStep * kStep);
  return state;
}

L4Planning::TrajectorySampler samplerFor(Plate plate)
{
  return [plate](double t) {
    L4Planning::AimState state;
    state.yaw = differentiate(plateYaw, plate, t);
    state.pitch = differentiate(platePitch, plate, t);
    return state;
  };
}

// ---- 1. 六个边界条件是否被逐条复现 ----
void testBoundaryConditions()
{
  const L4Planning::AxisState start{0.3, -1.2, 4.0};
  const L4Planning::AxisState end{-0.4, 0.9, -2.5};
  constexpr double kDuration = 0.137;

  const auto quintic = L4Planning::Quintic::fit(start, end, kDuration);

  checkNear(quintic.position(0.0), start.position, 1e-12, "p(0)");
  checkNear(quintic.velocity(0.0), start.velocity, 1e-12, "v(0)");
  checkNear(quintic.acceleration(0.0), start.acceleration, 1e-12, "a(0)");
  checkNear(quintic.position(kDuration), end.position, 1e-10, "p(T)");
  checkNear(quintic.velocity(kDuration), end.velocity, 1e-9, "v(T)");
  checkNear(quintic.acceleration(kDuration), end.acceleration, 1e-8, "a(T)");
  checkNear(quintic.duration, kDuration, 0.0, "duration recorded");
}

// ---- 2. 零边界速度/加速度时退化成教科书 S 曲线 ----
void testClassicSCurve()
{
  const auto quintic = L4Planning::Quintic::fit({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 1.0);

  // p = 10s^3 - 15s^4 + 6s^5
  checkNear(quintic.coefficient[3], 10.0, 1e-12, "S curve c3");
  checkNear(quintic.coefficient[4], -15.0, 1e-12, "S curve c4");
  checkNear(quintic.coefficient[5], 6.0, 1e-12, "S curve c5");
  checkNear(quintic.position(0.5), 0.5, 1e-12, "S curve midpoint");
  // 峰值加速度 10/sqrt(3) = 5.7735...，也就是 T = sqrt(5.774*delta/a_max)
  // 那个闭式估计的来源。
  checkNear(
    quintic.peakAbsAcceleration(), 10.0 / std::sqrt(3.0), 1e-9, "S curve peak accel");
}

// ---- 3. 解析峰值必须不低于稠密采样的峰值 ----
void testPeakAccelerationIsExact()
{
  const L4Planning::AxisState start{0.0, 2.0, -30.0};
  const L4Planning::AxisState end{0.15, -1.0, 12.0};
  constexpr double kDuration = 0.08;

  const auto quintic = L4Planning::Quintic::fit(start, end, kDuration);
  const double analytic = quintic.peakAbsAcceleration();

  double sampled = 0.0;
  constexpr int kSamples = 200000;
  for (int i = 0; i <= kSamples; ++i) {
    const double tau = kDuration * static_cast<double>(i) / kSamples;
    sampled = std::max(sampled, std::abs(quintic.acceleration(tau)));
  }

  check(analytic >= sampled - 1e-9, "analytic peak not below sampled peak");
  checkNear(analytic, sampled, 1e-4 * std::max(1.0, sampled), "analytic peak matches sampled");
}

// ---- 4. 退化时长必须自曝，不能看起来"最可行" ----
void testDegenerateDuration()
{
  const auto quintic = L4Planning::Quintic::fit({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, 0.0);
  check(quintic.duration == 0.0, "degenerate duration is zero");
  checkNear(quintic.position(0.0), 1.0, 0.0, "degenerate collapses to start");
  checkNear(quintic.peakAbsAcceleration(), 0.0, 0.0, "degenerate peak is zero");
}

// ---- 5. solveBlend 找到的是最小可行时长 ----
void testSolveBlendMinimality()
{
  const Plate before{3.0, 0.25, 10.0, 0.0, 0.0, 0.0};
  Plate after = before;
  after.phase = -std::numbers::pi / 2.0;
  after.height = 0.05;  // 让 pitch 也有一个台阶

  L4Planning::BlendLimits limits;
  limits.max_yaw_acceleration = 50.0;
  limits.max_pitch_acceleration = 100.0;

  const auto solution =
    L4Planning::solveBlend(samplerFor(before)(0.0), samplerFor(after), limits);

  check(solution.valid, "solveBlend returned a solution");
  check(!solution.acceleration_limited, "solved blend respects the accel limit");
  check(
    solution.peak_yaw_acceleration <= limits.max_yaw_acceleration,
    "solved yaw peak within limit");
  check(
    solution.duration > limits.min_duration,
    "this configuration really needs to search (min_duration insufficient)");

  // 再短一档就该超限，否则搜到的不是最小值。分辨率 = 区间 / 2^iterations。
  const double resolution =
    (limits.max_duration - limits.min_duration) /
    static_cast<double>(1 << limits.search_iterations);
  const auto shorter = L4Planning::fitBlend(
    samplerFor(before)(0.0), samplerFor(after), solution.duration - 2.0 * resolution,
    limits);
  check(shorter.valid, "shorter fit is well formed");
  check(shorter.acceleration_limited, "a shorter blend would exceed the limit");

  std::printf(
    "  [solveBlend] T = %.1f ms, peak yaw = %.1f rad/s^2, peak pitch = %.1f rad/s^2\n",
    solution.duration * 1e3, solution.peak_yaw_acceleration,
    solution.peak_pitch_acceleration);
}

// ---- 6. 能力不够时如实上报，不靠拉长过渡掩盖 ----
void testAccelerationLimitedIsReported()
{
  const Plate before{1.2, 0.28, 14.0, 0.0, 0.0, 0.0};
  Plate after = before;
  after.phase = -std::numbers::pi / 2.0;

  L4Planning::BlendLimits limits;
  limits.max_yaw_acceleration = 2.0;  // 明显不够的云台
  limits.max_pitch_acceleration = 2.0;

  const auto solution =
    L4Planning::solveBlend(samplerFor(before)(0.0), samplerFor(after), limits);
  check(solution.valid, "limited case still returns a solution");
  check(solution.acceleration_limited, "limited case is flagged");
  checkNear(solution.duration, limits.max_duration, 1e-12, "limited case uses max duration");
}

// ---- 7. 端到端：跟随段逐位不变、拼接连续、全程不超加速度 ----
void testSmootherEndToEnd()
{
  using Clock = L4Planning::AimSmoother::TimePoint::clock;

  const Plate before{3.0, 0.25, 10.0, 0.0, 0.0, std::numbers::pi - 0.03};
  Plate after = before;
  after.phase = -std::numbers::pi / 2.0;
  after.height = 0.05;

  L4Planning::BlendLimits limits;
  limits.max_yaw_acceleration = 50.0;
  limits.max_pitch_acceleration = 100.0;

  L4Planning::AimSmoother smoother(limits);

  constexpr double kSwitchTime = 0.30;  // 仿真起点到切板的时间
  constexpr double kStep = 1e-3;
  constexpr int kSteps = 500;

  const auto origin = Clock::now();
  std::vector<double> yaws;
  std::vector<double> times;
  bool ever_blended = false;
  bool blend_started = false;
  double blend_start_t = 0.0;
  double blend_end_t = 0.0;

  for (int i = 0; i < kSteps; ++i) {
    const double t = i * kStep;
    const bool switched = t >= kSwitchTime;
    const Plate& current = switched ? after : before;

    // 射击轨迹原值：切板前后分别是两块板。
    const double raw_yaw = plateYaw(current, t);
    const double raw_pitch = platePitch(current, t);

    std::optional<L4Planning::AimSmoother::Forecast> forecast;
    if (!switched) {
      L4Planning::AimSmoother::Forecast f;
      f.switch_time = kSwitchTime - t;
      // 采样器的时间原点是"本帧"，所以要把两块板的相位推到当前时刻。
      Plate shifted_before = before;
      Plate shifted_after = after;
      shifted_before.phase = before.phase + before.omega * t;
      shifted_after.phase = after.phase + after.omega * t;
      f.before = samplerFor(shifted_before)(0.0);
      f.after = samplerFor(shifted_after);
      forecast = std::move(f);
    }

    const auto out = smoother.update(
      origin + std::chrono::nanoseconds(static_cast<long long>(t * 1e9)), raw_yaw,
      raw_pitch, forecast);

    if (out.blending) {
      if (!blend_started) {
        blend_started = true;
        ever_blended = true;
        blend_start_t = t;
        // 起点必须与本帧正在下发的射击轨迹连续，否则过渡本身就是一次跳变。
        checkNear(
          wrapToPi(out.yaw - raw_yaw), 0.0, 1e-9, "blend starts continuous in yaw");
        checkNear(
          wrapToPi(out.pitch - raw_pitch), 0.0, 1e-9, "blend starts continuous in pitch");
      }
      blend_end_t = t;
      // late 现在是"晚出 commit_margin 以上"，帧量化那一两毫秒不算。
      check(!out.late, "blend committed with enough lead time");
    } else if (!blend_started) {
      // 过渡开始之前：跟随段必须逐位等于原值。
      check(out.yaw == raw_yaw, "pre-blend yaw is untouched");
      check(out.pitch == raw_pitch, "pre-blend pitch is untouched");
    } else {
      // 过渡结束之后同理。
      check(out.yaw == raw_yaw, "post-blend yaw is untouched");
      check(out.pitch == raw_pitch, "post-blend pitch is untouched");
    }

    times.push_back(t);
    yaws.push_back(out.yaw);
  }

  check(ever_blended, "a blend actually happened");
  // 终点必须落在切板时刻**或其之后**：提前结束的话，那时真正的切板还没
  // 发生，输出会先从新板轨迹掉回旧板、到切板时再跳一次，一个阶跃变成两个。
  // 晚的量只该是帧量化，触发条件是 switch_time <= 最小可行时长，一帧就能
  // 跨过等号。
  check(
    blend_end_t >= kSwitchTime - kStep,
    "blend must not finish before the switch instant");
  check(
    blend_end_t <= kSwitchTime + 4.0 * kStep,
    "blend overshoot past the switch must stay within frame quantisation");
  check(
    blend_start_t < kSwitchTime - 0.02,
    "blend starts meaningfully before the switch (anticipatory)");

  // 全程二阶差分：过渡段把阶跃磨平之后，整条命令序列的加速度都该在限内。
  double peak = 0.0;
  for (std::size_t i = 1; i + 1 < yaws.size(); ++i) {
    const double second_difference =
      (wrapToPi(yaws[i + 1] - yaws[i]) - wrapToPi(yaws[i] - yaws[i - 1])) /
      (kStep * kStep);
    peak = std::max(peak, std::abs(second_difference));
  }
  check(
    peak <= limits.max_yaw_acceleration * 1.05,
    "smoothed command respects the yaw acceleration limit end to end");

  std::printf(
    "  [end to end] blend %.0f -> %.0f ms (switch at %.0f ms), command peak accel"
    " = %.1f rad/s^2\n",
    blend_start_t * 1e3, blend_end_t * 1e3, kSwitchTime * 1e3, peak);

  // 同一条序列，不做平滑的话峰值是多少——用来确认这个用例真的有阶跃可磨。
  double raw_peak = 0.0;
  std::vector<double> raw;
  raw.reserve(kSteps);
  for (int i = 0; i < kSteps; ++i) {
    const double t = i * kStep;
    raw.push_back(plateYaw(t >= kSwitchTime ? after : before, t));
  }
  for (std::size_t i = 1; i + 1 < raw.size(); ++i) {
    const double second_difference =
      (wrapToPi(raw[i + 1] - raw[i]) - wrapToPi(raw[i] - raw[i - 1])) / (kStep * kStep);
    raw_peak = std::max(raw_peak, std::abs(second_difference));
  }
  check(raw_peak > limits.max_yaw_acceleration, "the raw trajectory really does exceed the limit");
  std::printf("  [end to end] unsmoothed peak accel = %.0f rad/s^2\n", raw_peak);
}

// ---- 8. reset 之后必须回到跟随段 ----
void testReset()
{
  using Clock = L4Planning::AimSmoother::TimePoint::clock;

  const Plate before{3.0, 0.25, 10.0, 0.0, 0.0, 0.0};
  Plate after = before;
  after.phase = -std::numbers::pi / 2.0;

  L4Planning::AimSmoother smoother;
  L4Planning::AimSmoother::Forecast forecast;
  forecast.switch_time = 0.05;
  forecast.before = samplerFor(before)(0.0);
  forecast.after = samplerFor(after);

  const auto now = Clock::now();
  const auto out = smoother.update(now, plateYaw(before, 0.0), 0.0, forecast);
  check(out.blending, "smoother commits when the switch is imminent");
  check(smoother.blending(), "blending() agrees");

  smoother.reset();
  check(!smoother.blending(), "reset clears the blend");

  const auto after_reset = smoother.update(now, 1.234, -0.5, std::nullopt);
  check(!after_reset.blending, "no forecast means no blend");
  check(after_reset.yaw == 1.234, "reset restores pass-through yaw");
  check(after_reset.pitch == -0.5, "reset restores pass-through pitch");
}

}  // namespace

int main()
{
  testBoundaryConditions();
  testClassicSCurve();
  testPeakAccelerationIsExact();
  testDegenerateDuration();
  testSolveBlendMinimality();
  testAccelerationLimitedIsReported();
  testSmootherEndToEnd();
  testReset();

  if (failures > 0) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "AimSmoother smoke test passed\n";
  return 0;
}
