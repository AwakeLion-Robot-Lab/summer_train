#include "l4_planning/tiny_mpc.hpp"

#include <chrono>
#include <cmath>
#include <utility>

namespace L4Planning {

namespace {

constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] double normalizeAngle(double angle) noexcept
{
  angle = std::remainder(angle, 2.0 * kPi);
  return angle <= -kPi ? angle + 2.0 * kPi : angle;
}

[[nodiscard]] TimePoint addSeconds(TimePoint time, double seconds)
{
  return time + std::chrono::duration_cast<TimePoint::duration>(
                  std::chrono::duration<double>(seconds));
}

}  // namespace

std::vector<AimSample> makeTinyMpcSamples(
  const TinyMpcReference& reference,
  const TinyMpcSolution& solution,
  TimePoint first_execute_time)
{
  std::vector<AimSample> samples;
  if (!reference.valid || !solution.valid) {
    return samples;
  }

  samples.reserve(kTinyMpcHorizon - kTinyMpcControlIndex);
  for (std::size_t index = kTinyMpcControlIndex;
       index < kTinyMpcHorizon;
       ++index) {
    AimSample sample;
    sample.execute_time = addSeconds(
      first_execute_time,
      static_cast<double>(index - kTinyMpcControlIndex)
        * kTinyMpcStepSeconds);
    sample.yaw = normalizeAngle(
      reference.yaw_origin + solution.yaw.angle[index]);
    sample.pitch = solution.pitch.angle[index];
    sample.yaw_rate = solution.yaw.angular_velocity[index];
    sample.pitch_rate = solution.pitch.angular_velocity[index];
    sample.yaw_acceleration = solution.yaw.acceleration[index];
    sample.pitch_acceleration = solution.pitch.acceleration[index];
    if (index > 0) {
      sample.yaw_jerk =
        (solution.yaw.acceleration[index]
         - solution.yaw.acceleration[index - 1])
        / kTinyMpcStepSeconds;
      sample.pitch_jerk =
        (solution.pitch.acceleration[index]
         - solution.pitch.acceleration[index - 1])
        / kTinyMpcStepSeconds;
    }
    if (!std::isfinite(sample.yaw) || !std::isfinite(sample.pitch)
        || !std::isfinite(sample.yaw_rate)
        || !std::isfinite(sample.pitch_rate)
        || !std::isfinite(sample.yaw_acceleration)
        || !std::isfinite(sample.pitch_acceleration)
        || !std::isfinite(sample.yaw_jerk)
        || !std::isfinite(sample.pitch_jerk)) {
      samples.clear();
      return samples;
    }
    samples.push_back(sample);
  }
  return samples;
}

AimPlan applyTinyMpc(
  const AimPlan& direct_plan,
  const L3Estimation::TargetState& target,
  const L1Sensor::RobotState& robot_state,
  const Eigen::Isometry3d& T_barrel_world,
  const PlannerConfig& config)
{
  AimPlan result = direct_plan;
  result.using_MPC = false;
  result.samples.clear();
  if (!config.enable_mpc || !direct_plan.valid) {
    return result;
  }

  const TinyMpcReference reference = buildTinyMpcReference(
    direct_plan, target, robot_state, T_barrel_world, config);
  const TinyMpcSolution solution = solveTinyMpc(reference, config);
  std::vector<AimSample> samples = makeTinyMpcSamples(
    reference, solution, direct_plan.generated_at);
  if (samples.empty()) {
    return result;
  }

  result.samples = std::move(samples);
  result.using_MPC = true;
  const AimSample& current = result.samples.front();
  result.yaw = current.yaw;
  result.pitch = current.pitch;
  result.yaw_rate = current.yaw_rate;
  result.pitch_rate = current.pitch_rate;
  result.yaw_acceleration = current.yaw_acceleration;
  result.pitch_acceleration = current.pitch_acceleration;
  return result;
}

}  // namespace L4Planning
