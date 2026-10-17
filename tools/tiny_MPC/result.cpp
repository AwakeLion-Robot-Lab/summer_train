#include "l3_planning/tiny_mpc.hpp"

#include <utility>

namespace L4Planning {

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
