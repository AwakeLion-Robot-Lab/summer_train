#include "l4_planning/aim_smoother.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l6_telemetry/quintic_trace.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>

namespace {
using namespace L4Planning;
void require(bool ok, const char* message)
{
  if (!ok) throw std::runtime_error(message);
}
void near(double a, double b, double tolerance, const char* message)
{
  require(std::abs(a - b) <= tolerance, message);
}
TimePoint at(double seconds)
{
  return TimePoint{} + std::chrono::duration_cast<TimePoint::duration>(
    std::chrono::duration<double>{seconds});
}
AimState state(double p, double v = 0.0, double a = 0.0)
{
  return {{p, v, a}, {0.0, 0.0, 0.0}};
}
Plan plan(double p)
{
  Plan result;
  result.status = PlanStatus::TrackOnly;
  result.aim.yaw = result.aim.shoot_yaw = p;
  result.fire = FireReference{};
  return result;
}
PlannerDiagnostics tracking(double time, double bias = 0.0)
{
  PlannerDiagnostics d;
  d.raw_valid = true;
  d.raw_trajectory = [time, bias](double offset) {
    return state(0.8 * (time + offset) + bias, 0.8);
  };
  d.raw = d.raw_trajectory(0.0);
  return d;
}

void velocityPeaks()
{
  const auto s = Quintic::fit({0, 0, 0}, {1, 0, 0}, 1);
  near(s.peakAbsVelocity(), 1.875, 1e-12, "S curve peak speed");
  near(Quintic::fit({1, -3, 0}, {-5, -3, 0}, 2).peakAbsVelocity(),
    3, 1e-12, "constant negative speed");
  near(Quintic::fit({0, 0, 2}, {1, 2, 2}, 1).peakAbsVelocity(),
    2, 1e-12, "quadratic position endpoint speed");
  std::mt19937 generator(4711);
  std::uniform_real_distribution<double> random(-1.0, 1.0);
  for (int sample = 0; sample < 80; ++sample) {
    const double duration = std::pow(10.0, -3.0 + 4.0 * (random(generator) + 1.0) / 2.0);
    const auto q = Quintic::fit(
      {random(generator), random(generator) * 4, random(generator) * 30},
      {random(generator), random(generator) * 4, random(generator) * 30}, duration);
    double sampled_v = 0, sampled_a = 0;
    for (int i = 0; i <= 20000; ++i) {
      const double t = duration * i / 20000.0;
      sampled_v = std::max(sampled_v, std::abs(q.velocity(t)));
      sampled_a = std::max(sampled_a, std::abs(q.acceleration(t)));
    }
    const double exact_v = q.peakAbsVelocity();
    const double exact_a = q.peakAbsAcceleration();
    require(exact_v >= sampled_v - 1e-7 * std::max(1.0, exact_v), "speed peak underestimated");
    require(exact_a >= sampled_a - 1e-7 * std::max(1.0, exact_a), "acc peak underestimated");
    near(exact_v, sampled_v, 1e-5 * std::max(1.0, exact_v), "speed peak dense cross check");
  }
}

void replanAndMissingData()
{
  L6Telemetry::QuinticTrace trace({}, 10, 10);
  auto data = trace.update(at(0), plan(0), tracking(0), 0, 0);
  data = trace.update(at(0.013), plan(0.0104), tracking(0.013), 0, 0);
  near(data["replan"]["yaw_position_rad"], 0, 1e-12, "normal motion is not replan jump");
  data = trace.update(at(0.03), plan(0.074), tracking(0.03, 0.05), 0, 0);
  near(data["replan"]["yaw_position_rad"], 0.05, 1e-12, "changed prediction exposed");
  data = trace.update(at(0.05), Plan{}, PlannerDiagnostics{}, 0, 0);
  require(data["raw_valid"] == 0, "missing raw reference invalid");
  const auto wire = nlohmann::json::parse(data.dump());
  require(wire["yaw"]["planned_rad"].is_null(), "missing angles serialize to null");
  trace.reset();
  data = trace.update(at(0.08), plan(0.064), tracking(0.08), 0, 0);
  require(nlohmann::json::parse(data.dump())["yaw"]["planned_fd_velocity_rad_s"].is_null(),
    "reset must break derivative history");
}

void boundariesAndFailure()
{
  BlendLimits limits;
  limits.max_yaw_acceleration = 2;
  L6Telemetry::QuinticTrace trace(limits, 1, 1);
  const auto after = [](double) { return state(1); };
  const auto segment = fitBlend(state(0), after, 0.1, limits);
  require(segment.valid && !segment.feasible(), "numeric validity differs from feasibility");
  PlannerDiagnostics d;
  d.raw_valid = true;
  d.raw = state(0);
  d.raw_trajectory = [](double) { return state(0); };
  d.segment = segment;
  d.segment_start = at(0);
  d.forecast = AimSmoother::Forecast{0.1, state(0), after};
  d.smoother.committed = d.smoother.search_attempted = true;
  d.smoother.candidate = segment;
  auto p = plan(0);
  p.aim.blending = true;
  auto data = trace.update(at(0), p, d, 0, 0);
  require(data["blend"]["segment_acc_feasible"] == 0, "infeasible segment exposed");
  require(data["blend"]["search_acc_feasible"] == 0, "failed search exposed");
  require(data["yaw"]["segment_speed_exceeded"] == 1, "speed violation exposed");
  near(data["boundary"]["start"]["yaw_position_rad"], 0, 1e-12, "start boundary residual");
  near(data["boundary"]["end"]["yaw_acceleration_rad_s2"], 0, 1e-8, "end boundary residual");
  d = {};
  d.raw_valid = true;
  d.raw_trajectory = [](double) { return state(1.02); };
  d.raw = state(1.02);
  data = trace.update(at(0.1), plan(1.02), d, 0, 0);
  require(data["join"]["end_event"] == 1 && data["join"]["end_valid"] == 1,
    "completion join event exposed");
  near(data["join"]["end"]["yaw_position_rad"], 0.02, 1e-12, "live endpoint drift exposed");
  near(data["coverage"]["blend_s"], 0.1, 1e-12, "exact blend time integration");
}

void plannerDiagnosticsPreserveCommands()
{
  ArmorPlanConfig config;
  config.blend.enable = true;
  config.impact.send_to_control = 0;
  config.impact.high_speed_delay_time = config.impact.low_speed_delay_time = 0;
  Planner baseline(config), observed(config);
  observed.enableDiagnostics(true);
  L3Estimation::TrackedTarget original(
    L3Estimation::ArmorName::Infantry3, 4.0, 6.0, 0.26);
  original.jumped = true;
  int blend_frames = 0;
  for (int i = 0; i < 700; ++i) {
    PlanInput input;
    input.target = original;
    input.plan_time = at(i * 0.005);
    input.target->predict(input.plan_time);
    input.robot_state.bullet_speed = 25;
    const auto a = baseline.plan(input);
    const auto b = observed.plan(input);
    require(a.valid() == b.valid() && a.aim.yaw == b.aim.yaw &&
      a.aim.pitch == b.aim.pitch && a.aim.blending == b.aim.blending,
      "diagnostics changed planner behavior");
    if (b.aim.blending) {
      ++blend_frames;
      require(observed.diagnostics().segment.valid, "active blend diagnostic absent");
    }
  }
  require(blend_frames > 0, "integration must exercise real planner switches");
}
}  // namespace

int main()
{
  try {
    velocityPeaks();
    replanAndMissingData();
    boundariesAndFailure();
    plannerDiagnosticsPreserveCommands();
    std::cout << "Quintic trace smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
