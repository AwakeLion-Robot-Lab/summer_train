#include "l4_planning/tiny_mpc.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace L4Planning {

namespace {

[[nodiscard]] bool validAxisConfig(
  const TinyMpcAxisConfig& config) noexcept
{
  return std::isfinite(config.angle_weight)
         && config.angle_weight >= 0.0
         && std::isfinite(config.velocity_weight)
         && config.velocity_weight >= 0.0
         && std::isfinite(config.acceleration_weight)
         && config.acceleration_weight > 0.0
         && std::isfinite(config.min_acceleration)
         && std::isfinite(config.max_acceleration)
         && config.min_acceleration <= config.max_acceleration
         && std::isfinite(config.rho)
         && config.rho > 0.0
         && config.max_iterations > 0
         && std::isfinite(config.primal_tolerance)
         && config.primal_tolerance > 0.0
         && std::isfinite(config.dual_tolerance)
         && config.dual_tolerance > 0.0;
}

using State = Eigen::Vector2d;
using StateMatrix = Eigen::Matrix2d;
using Gain = Eigen::RowVector2d;

constexpr std::size_t kControlCount = kTinyMpcHorizon - 1;

[[nodiscard]] bool finiteReference(
  const TinyMpcAxisReference& reference) noexcept
{
  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    if (!std::isfinite(reference.angle[index])
        || !std::isfinite(reference.angular_velocity[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double automaticRho(const TinyMpcAxisConfig& config) noexcept
{
  (void)config;
  // 与 README 所描述的参考实现保持一致。调用者仍可通过配置显式覆盖。
  return 1.0;
}

}  // namespace

TinyMpcAxisSolution solveTinyMpcAxis(
  const TinyMpcAxisReference& reference,
  const TinyMpcAxisConfig& requested_config)
{
  TinyMpcAxisSolution solution;
  TinyMpcAxisConfig config = requested_config;
  if (config.rho == 0.0) {
    config.rho = automaticRho(config);
  }
  if (!validAxisConfig(config) || !finiteReference(reference)) {
    return solution;
  }

  const StateMatrix A = (StateMatrix() <<
    1.0, kTinyMpcStepSeconds,
    0.0, 1.0).finished();
  const State B{0.0, kTinyMpcStepSeconds};
  const StateMatrix Q = (StateMatrix() <<
    config.angle_weight, 0.0,
    0.0, config.velocity_weight).finished();
  std::array<StateMatrix, kTinyMpcHorizon> value_hessian{};
  std::array<Gain, kControlCount> feedback{};
  std::array<double, kControlCount> inverse_control_hessian{};
  const auto precompute = [&]() {
    const double augmented_r = config.acceleration_weight + config.rho;
    value_hessian.back() = Q;
    for (std::size_t reverse = kControlCount; reverse-- > 0;) {
      const StateMatrix& next = value_hessian[reverse + 1];
      const double quu = augmented_r + (B.transpose() * next * B).value();
      if (!std::isfinite(quu) || quu <= 0.0) {
        return false;
      }
      inverse_control_hessian[reverse] = 1.0 / quu;
      feedback[reverse] =
        inverse_control_hessian[reverse] * B.transpose() * next * A;
      const StateMatrix closed_loop = A - B * feedback[reverse];
      value_hessian[reverse] =
        Q + A.transpose() * next * closed_loop;
    }
    return true;
  };
  if (!precompute()) {
    return solution;
  }

  std::array<double, kControlCount> slack{};
  std::array<double, kControlCount> previous_slack{};
  std::array<double, kControlCount> dual{};
  std::array<double, kControlCount> dynamics_control{};
  std::array<double, kControlCount> feedforward{};
  std::array<State, kTinyMpcHorizon> dynamics_state{};
  std::array<State, kTinyMpcHorizon> value_gradient{};

  double primal_residual = std::numeric_limits<double>::infinity();
  double dual_residual = std::numeric_limits<double>::infinity();
  for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
    value_gradient.back() = -Q * State{
      reference.angle.back(), reference.angular_velocity.back()};
    for (std::size_t reverse = kControlCount; reverse-- > 0;) {
      const double control_linear =
        config.rho * (dual[reverse] - slack[reverse]);
      feedforward[reverse] = inverse_control_hessian[reverse]
        * (B.dot(value_gradient[reverse + 1]) + control_linear);
      value_gradient[reverse] =
        -Q * State{reference.angle[reverse],
                   reference.angular_velocity[reverse]}
        + (A - B * feedback[reverse]).transpose()
            * value_gradient[reverse + 1]
        - feedback[reverse].transpose() * control_linear;
    }

    dynamics_state[0] = {
      reference.angle[0], reference.angular_velocity[0]};
    for (std::size_t index = 0; index < kControlCount; ++index) {
      dynamics_control[index] =
        -feedback[index].dot(dynamics_state[index]) - feedforward[index];
      dynamics_state[index + 1] =
        A * dynamics_state[index] + B * dynamics_control[index];
    }

    previous_slack = slack;
    primal_residual = 0.0;
    dual_residual = 0.0;
    for (std::size_t index = 0; index < kControlCount; ++index) {
      slack[index] = std::clamp(
        dynamics_control[index] + dual[index],
        config.min_acceleration,
        config.max_acceleration);
      dual[index] += dynamics_control[index] - slack[index];
      primal_residual = std::max(
        primal_residual,
        std::abs(dynamics_control[index] - slack[index]));
      dual_residual = std::max(
        dual_residual,
        config.rho * std::abs(slack[index] - previous_slack[index]));
    }

    solution.info.iteration_count = iteration;
    solution.info.primal_residual = primal_residual;
    solution.info.dual_residual = dual_residual;
    if (primal_residual <= config.primal_tolerance
        && dual_residual <= config.dual_tolerance) {
      solution.info.converged = true;
      break;
    }

    // 残差相差过大时按 ADMM residual balancing 调整 rho。缩放对偶变量
    // 同步换算，随后重新预计算 Riccati 矩阵。
    if (iteration % 10 == 0) {
      double next_rho = config.rho;
      if (primal_residual > 10.0 * dual_residual) {
        next_rho = std::min(config.rho * 2.0, 1e6);
      } else if (dual_residual > 10.0 * primal_residual) {
        next_rho = std::max(config.rho * 0.5, 1e-6);
      }
      if (next_rho != config.rho) {
        const double dual_scale = config.rho / next_rho;
        for (double& value : dual) {
          value *= dual_scale;
        }
        config.rho = next_rho;
        if (!precompute()) {
          return solution;
        }
      }
    }
  }

  // ADMM 的 dynamics_control 在未收敛时可能略微越界。用投影后的控制量
  // 重新正向展开，保证返回轨迹严格满足盒约束和离散动力学。
  State state{reference.angle[0], reference.angular_velocity[0]};
  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    solution.angle[index] = state.x();
    solution.angular_velocity[index] = state.y();
    if (index < kControlCount) {
      solution.acceleration[index] = slack[index];
      state = A * state + B * slack[index];
    }
  }
  solution.acceleration.back() = solution.acceleration[kControlCount - 1];

  solution.valid = solution.info.converged;
  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    solution.valid = solution.valid
      && std::isfinite(solution.angle[index])
      && std::isfinite(solution.angular_velocity[index])
      && std::isfinite(solution.acceleration[index]);
  }
  return solution;
}

TinyMpcSolution solveTinyMpc(
  const TinyMpcReference& reference,
  const PlannerConfig& config)
{
  TinyMpcSolution solution;
  if (!reference.valid) {
    return solution;
  }

  const TinyMpcAxisConfig yaw_config{
    config.yaw_angle_weight,
    config.yaw_velocity_weight,
    config.yaw_acceleration_weight,
    config.min_yaw_acceleration,
    config.max_yaw_acceleration,
    config.mpc_admm_rho,
    config.mpc_max_iterations,
    config.mpc_primal_tolerance,
    config.mpc_dual_tolerance};
  const TinyMpcAxisConfig pitch_config{
    config.pitch_angle_weight,
    config.pitch_velocity_weight,
    config.pitch_acceleration_weight,
    config.min_pitch_acceleration,
    config.max_pitch_acceleration,
    config.mpc_admm_rho,
    config.mpc_max_iterations,
    config.mpc_primal_tolerance,
    config.mpc_dual_tolerance};

  solution.yaw = solveTinyMpcAxis(reference.yaw, yaw_config);
  solution.pitch = solveTinyMpcAxis(reference.pitch, pitch_config);
  solution.valid = solution.yaw.valid && solution.pitch.valid;
  return solution;
}

}  // namespace L4Planning
