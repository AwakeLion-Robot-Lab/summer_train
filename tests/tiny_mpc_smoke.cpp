#include "l4_planning/tiny_mpc.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace {

bool close(double left, double right, double tolerance = 1e-8)
{
  return std::abs(left - right) <= tolerance;
}

L4Planning::TinyMpcAxisConfig testConfig()
{
  return {
    .angle_weight = 10000.0,
    .velocity_weight = 10.0,
    .acceleration_weight = 1.0,
    .min_acceleration = -3.0,
    .max_acceleration = 3.0,
    .rho = 0.0,
    .max_iterations = 1000,
    .primal_tolerance = 1e-3,
    .dual_tolerance = 1e-3};
}

}  // namespace

int main()
{
  using namespace L4Planning;
  using namespace std::chrono_literals;

  TinyMpcAxisReference steady;
  steady.angle.fill(0.4);
  steady.angular_velocity.fill(0.0);
  const TinyMpcAxisSolution steady_solution =
    solveTinyMpcAxis(steady, testConfig());
  if (!steady_solution.valid || !steady_solution.info.converged) {
    std::cerr << "steady reference did not converge\n";
    return 1;
  }
  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    if (!close(steady_solution.angle[index], 0.4)
        || !close(steady_solution.angular_velocity[index], 0.0)
        || !close(steady_solution.acceleration[index], 0.0)) {
      std::cerr << "steady reference was not preserved\n";
      return 2;
    }
  }

  TinyMpcAxisReference constrained;
  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    constrained.angle[index] = index < 30 ? 0.0 : 2.0;
    constrained.angular_velocity[index] = 0.0;
  }
  const TinyMpcAxisSolution constrained_solution =
    solveTinyMpcAxis(constrained, testConfig());
  if (!constrained_solution.valid) {
    std::cerr << "constrained reference did not converge: primal="
              << constrained_solution.info.primal_residual
              << " dual=" << constrained_solution.info.dual_residual << '\n';
    return 3;
  }
  for (std::size_t index = 0; index + 1 < kTinyMpcHorizon; ++index) {
    const double acceleration = constrained_solution.acceleration[index];
    if (acceleration < -3.0 - 1e-12 || acceleration > 3.0 + 1e-12) {
      std::cerr << "acceleration bound was violated\n";
      return 4;
    }
    const double expected_angle = constrained_solution.angle[index]
      + kTinyMpcStepSeconds
          * constrained_solution.angular_velocity[index];
    const double expected_velocity =
      constrained_solution.angular_velocity[index]
      + kTinyMpcStepSeconds * acceleration;
    if (!close(constrained_solution.angle[index + 1], expected_angle)
        || !close(constrained_solution.angular_velocity[index + 1],
                  expected_velocity)) {
      std::cerr << "discrete dynamics were violated\n";
      return 5;
    }
  }

  L3Estimation::TargetState wrapping_target;
  wrapping_target.robot_id = 3;
  wrapping_target.center = {-5.0, -0.5, 0.0};
  wrapping_target.velocity = {0.0, 1.0, 0.0};
  wrapping_target.yaw = 0.0;
  wrapping_target.radius = 0.25;
  wrapping_target.covariance =
    L3Estimation::StateCovariance::Identity() * 1e-3;
  wrapping_target.timestamp = TimePoint{2s};
  L1Sensor::RobotState wrapping_robot;
  wrapping_robot.bullet_speed = 30.0;
  wrapping_robot.timestamp = wrapping_target.timestamp;
  AimPlan wrapping_direct;
  wrapping_direct.valid = true;
  wrapping_direct.armor_id = 0;
  wrapping_direct.yaw = 3.14159265358979323846;
  wrapping_direct.impact_time = TimePoint{2500ms};
  const TinyMpcReference wrapping_reference = buildTinyMpcReference(
    wrapping_direct,
    wrapping_target,
    wrapping_robot,
    Eigen::Isometry3d::Identity(),
    PlannerConfig{});
  if (!wrapping_reference.valid) {
    std::cerr << "yaw wrapping reference was not generated\n";
    return 6;
  }
  for (std::size_t index = 1; index < kTinyMpcHorizon; ++index) {
    if (std::abs(wrapping_reference.yaw.angle[index]
                 - wrapping_reference.yaw.angle[index - 1]) > 0.1
        || std::abs(wrapping_reference.yaw.angular_velocity[index]) > 10.0) {
      std::cerr << "yaw wrapping introduced a discontinuity\n";
      return 7;
    }
  }

  TinyMpcReference output_reference;
  output_reference.valid = true;
  output_reference.yaw_origin = 3.13;
  TinyMpcSolution output_solution;
  output_solution.valid = true;
  output_solution.yaw.valid = true;
  output_solution.pitch.valid = true;
  output_solution.yaw.angle.fill(0.03);
  output_solution.pitch.angle.fill(0.1);
  const TimePoint start{1s};
  const std::vector<AimSample> samples = makeTinyMpcSamples(
    output_reference, output_solution, start);
  if (samples.size() != 50 || samples.front().execute_time != start
      || samples.back().execute_time != start + 490ms
      || samples.front().yaw <= -3.14159265358979323846
      || samples.front().yaw > 3.14159265358979323846) {
    std::cerr << "MPC output protocol is invalid\n";
    return 8;
  }

  Planner planner;
  PlannerContext context;
  context.config.lock_stable_frames = 1;
  context.config.switch_dead_zone = 180.0;
  context.config.mpc_max_iterations = 10;
  context.planning_time = TimePoint{2s};
  L1Sensor::RobotState robot;
  robot.timestamp = context.planning_time;
  robot.bullet_speed = 30.0;
  L3Estimation::TargetState target;
  target.robot_id = 3;
  target.center = {5.0, 0.0, 0.0};
  target.velocity = {0.0, 0.2, 0.0};
  target.yaw = 0.0;
  target.yaw_rate = 0.5;
  target.radius = 0.25;
  target.covariance = L3Estimation::StateCovariance::Identity() * 1e-3;
  target.timestamp = context.planning_time;
  const AimPlan integrated = planner.plan(target, robot, context);
  if (!integrated.valid || !integrated.using_MPC
      || integrated.samples.size() != 50
      || integrated.samples.front().execute_time != integrated.generated_at) {
    const TinyMpcReference integrated_reference = buildTinyMpcReference(
      integrated, target, robot, context.T_barrel_world, context.config);
    const TinyMpcSolution integrated_solution = solveTinyMpc(
      integrated_reference, context.config);
    std::cerr << "Planner did not publish a valid MPC trajectory: valid="
              << integrated.valid << " using=" << integrated.using_MPC
              << " samples=" << integrated.samples.size()
              << " reference=" << integrated_reference.valid
              << " yaw=" << integrated_solution.yaw.valid
              << " yaw_primal=" << integrated_solution.yaw.info.primal_residual
              << " yaw_dual=" << integrated_solution.yaw.info.dual_residual
              << " pitch=" << integrated_solution.pitch.valid
              << " pitch_primal=" << integrated_solution.pitch.info.primal_residual
              << " pitch_dual=" << integrated_solution.pitch.info.dual_residual
              << '\n';
    return 9;
  }

  L5Control::FireDecision decision;
  decision.shoot = true;
  const L5Control::SerialCommand command =
    L5Control::Controller{}.makeCommand(integrated, decision, robot);
  const AimSample& current = integrated.samples.front();
  if (!close(command.yaw, current.yaw)
      || !close(command.pitch, current.pitch)
      || !close(command.yaw_rate, current.yaw_rate)
      || !close(command.pitch_rate, current.pitch_rate)
      || !close(command.yaw_acceleration, current.yaw_acceleration)
      || !close(command.pitch_acceleration, current.pitch_acceleration)
      || !command.shoot) {
    std::cerr << "Controller dropped MPC motion fields\n";
    return 10;
  }

  std::cout << "TinyMPC smoke test passed\n";
  return 0;
}
