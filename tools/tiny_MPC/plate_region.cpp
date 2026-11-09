#include "l3_planning/tiny_mpc.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace L4Planning {

Eigen::Vector2d projectTinyMpcPlate(
  const TinyMpcPlateRegion& region, const Eigen::Vector2d& angles,
  const Eigen::Vector2d& weights)
{
  const Eigen::Matrix2d h = region.jacobian.transpose() * weights.asDiagonal() * region.jacobian;
  const Eigen::Vector2d b = region.jacobian.transpose() * weights.asDiagonal() * (angles - region.center);
  Eigen::Vector2d best = Eigen::Vector2d::Zero();
  double cost = std::numeric_limits<double>::infinity();
  const auto consider = [&](Eigen::Vector2d uv) {
    uv = uv.cwiseMax(-region.limits).cwiseMin(region.limits);
    const Eigen::Vector2d error = region.center + region.jacobian * uv - angles;
    const double candidate_cost = error.dot(weights.asDiagonal() * error);
    if (candidate_cost < cost) { cost = candidate_cost; best = uv; }
  };
  consider(best);
  if (h.determinant() > 1e-12 * h.trace() * h.trace()) {
    const Eigen::Vector2d unconstrained = h.ldlt().solve(b);
    if ((unconstrained.array().abs() <= region.limits.array()).all()) consider(unconstrained);
  }
  // 二维凸二次函数在矩形上的最优点必在内部或某条边上。
  for (int fixed = 0; fixed < 2; ++fixed) {
    const int free = 1 - fixed;
    for (const double sign : {-1.0, 1.0}) {
      Eigen::Vector2d uv = Eigen::Vector2d::Zero();
      uv[fixed] = sign * region.limits[fixed];
      uv[free] = h(free, free) > 1e-20
        ? (b[free] - h(free, fixed) * uv[fixed]) / h(free, free) : 0.0;
      consider(uv);
    }
  }
  return best;
}

TinyMpcSolution solveTinyMpcPlateRegion(
  const TinyMpcReference& reference, const PlannerConfig& config)
{
  TinyMpcSolution result;
  if (!reference.valid || !reference.optimize_in_plate || config.mpc_region_max_iterations <= 0
      || !std::isfinite(config.mpc_admm_rho) || config.mpc_admm_rho <= 0.0
      || !std::isfinite(config.mpc_primal_tolerance) || config.mpc_primal_tolerance <= 0.0
      || !std::isfinite(config.mpc_dual_tolerance) || config.mpc_dual_tolerance <= 0.0) return result;

  const Eigen::Vector2d weight{config.yaw_angle_weight, config.pitch_angle_weight};
  const Eigen::Vector2d velocity_weight{config.yaw_velocity_weight, config.pitch_velocity_weight};
  const Eigen::Vector2d control_weight{config.yaw_acceleration_weight, config.pitch_acceleration_weight};
  const Eigen::Vector2d lower{config.min_yaw_acceleration, config.min_pitch_acceleration};
  const Eigen::Vector2d upper{config.max_yaw_acceleration, config.max_pitch_acceleration};
  if (!weight.allFinite() || !velocity_weight.allFinite() || !control_weight.allFinite()
      || !lower.allFinite() || !upper.allFinite() || (weight.array() < 0.0).any()
      || (velocity_weight.array() < 0.0).any() || (control_weight.array() <= 0.0).any()
      || (lower.array() > upper.array()).any()) return result;
  for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
    const auto& region = reference.regions[k];
    if (!region.center.allFinite() || !region.jacobian.allFinite() || !region.limits.allFinite()
        || (region.limits.array() < 0.0).any()
        || !std::isfinite(reference.yaw.angle[k]) || !std::isfinite(reference.pitch.angle[k])
        || !std::isfinite(reference.yaw.angular_velocity[k])
        || !std::isfinite(reference.pitch.angular_velocity[k])) return result;
  }

  // min sum distance_W(theta, ballistic(plate rectangle))^2 + velocity_Q^2 + acceleration_R^2.
  // 角度和 u/v 通过 ADMM 联合优化。板内变量严格有界；板间空隙允许跟踪残差，
  // 当前控制点必须再通过非线性弹道/运动板面交点验证才有射击资格。
  constexpr std::size_t controls = kTinyMpcHorizon - 1;
  using VectorArray = std::array<Eigen::Vector2d, kTinyMpcHorizon>;
  VectorArray angle{}, z{}, angle_dual{}, slack{}, control_dual{}, acceleration{};
  // 跟踪权重远大于控制权重。对角度分裂取平方根尺度，避免板内自由方向
  // 被巨大的 ADMM 惩罚锁住，导致位置几乎不动却提前满足残差阈值。
  Eigen::Vector2d rho_angle = weight.cwiseMax(1.0).cwiseSqrt();
  double rho_control = config.mpc_admm_rho;
  const Eigen::Matrix2d a = (Eigen::Matrix2d() << 1.0, kTinyMpcStepSeconds, 0.0, 1.0).finished();
  const Eigen::Vector2d b{0.0, kTinyMpcStepSeconds};
  struct AxisCache {
    std::array<Eigen::Matrix2d, kTinyMpcHorizon> p{};
    std::array<Eigen::RowVector2d, controls> gain{};
    std::array<double, controls> inverse{};
  };
  std::array<AxisCache, 2> cache;
  const auto precompute = [&]() {
    for (int axis = 0; axis < 2; ++axis) {
      auto& c = cache[axis];
      const Eigen::Matrix2d q = Eigen::Vector2d{rho_angle[axis], velocity_weight[axis]}.asDiagonal();
      c.p.back() = q;
      for (std::size_t k = controls; k-- > 0;) {
        const double denominator = control_weight[axis] + rho_control + b.dot(c.p[k + 1] * b);
        if (!std::isfinite(denominator) || denominator <= 0.0) return false;
        c.inverse[k] = 1.0 / denominator;
        c.gain[k] = c.inverse[k] * b.transpose() * c.p[k + 1] * a;
        c.p[k] = q + a.transpose() * c.p[k + 1] * (a - b * c.gain[k]);
      }
    }
    return true;
  };
  if (!precompute()) return result;
  for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
    z[k] = {reference.yaw.angle[k], reference.pitch.angle[k]};
    angle[k] = z[k];
    angle_dual[k].setZero();
    slack[k].setZero(); control_dual[k].setZero(); acceleration[k].setZero();
  }

  TinyMpcSolveInfo info;
  for (int iteration = 1; iteration <= config.mpc_region_max_iterations; ++iteration) {
    for (int axis = 0; axis < 2; ++axis) {
      const auto& ref = axis == 0 ? reference.yaw : reference.pitch;
      const auto& c = cache[axis];
      std::array<Eigen::Vector2d, kTinyMpcHorizon> gradient{};
      std::array<double, controls> feedforward{};
      const auto linear = [&](std::size_t k) -> Eigen::Vector2d {
        return {-rho_angle[axis] * (z[k][axis] - angle_dual[k][axis]),
                -velocity_weight[axis] * ref.angular_velocity[k]};
      };
      gradient.back() = linear(kTinyMpcHorizon - 1);
      for (std::size_t k = controls; k-- > 0;) {
        const double control_linear = rho_control * (control_dual[k][axis] - slack[k][axis]);
        feedforward[k] = c.inverse[k] * (b.dot(gradient[k + 1]) + control_linear);
        gradient[k] = linear(k) + (a - b * c.gain[k]).transpose() * gradient[k + 1]
          - c.gain[k].transpose() * control_linear;
      }
      // 沿用 TinyMPC 的居中窗口：窗口首端固定，执行第 50 点。
      Eigen::Vector2d state{ref.angle.front(), ref.angular_velocity.front()};
      for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
        angle[k][axis] = state.x();
        if (k < controls) {
          acceleration[k][axis] = -c.gain[k].dot(state) - feedforward[k];
          state = a * state + b * acceleration[k][axis];
        }
      }
    }
    info.primal_residual = info.dual_residual = 0.0;
    Eigen::Vector2d angle_primal = Eigen::Vector2d::Zero(), angle_dual_residual = Eigen::Vector2d::Zero();
    double control_primal = 0.0, control_dual_residual = 0.0;
    const Eigen::Vector2d projection_weight =
      (weight.array() * rho_angle.array() / (weight + rho_angle).array()).matrix();
    for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
      const Eigen::Vector2d old_z = z[k];
      const Eigen::Vector2d query = angle[k] + angle_dual[k];
      result.in_plate_offsets[k] = projectTinyMpcPlate(reference.regions[k], query, projection_weight);
      const Eigen::Vector2d projected = reference.regions[k].center
        + reference.regions[k].jacobian * result.in_plate_offsets[k];
      z[k] = ((weight.array() * projected.array() + rho_angle.array() * query.array())
        / (weight + rho_angle).array()).matrix();
      angle_dual[k] += angle[k] - z[k];
      angle_primal = angle_primal.cwiseMax((angle[k] - z[k]).cwiseAbs());
      angle_dual_residual = angle_dual_residual.cwiseMax((z[k] - old_z).cwiseAbs());
      info.primal_residual = std::max(info.primal_residual, (angle[k] - z[k]).lpNorm<Eigen::Infinity>());
      // 角度一致性的对偶残差按 rho_angle 归一化，以 rad 与配置阈值比较。
      info.dual_residual = std::max(info.dual_residual, (z[k] - old_z).lpNorm<Eigen::Infinity>());
      if (k < controls) {
        const Eigen::Vector2d old_slack = slack[k];
        slack[k] = (acceleration[k] + control_dual[k]).cwiseMax(lower).cwiseMin(upper);
        control_dual[k] += acceleration[k] - slack[k];
        control_primal = std::max(control_primal, (acceleration[k] - slack[k]).lpNorm<Eigen::Infinity>());
        control_dual_residual = std::max(control_dual_residual,
          rho_control * (slack[k] - old_slack).lpNorm<Eigen::Infinity>());
        info.primal_residual = std::max(info.primal_residual,
          (acceleration[k] - slack[k]).lpNorm<Eigen::Infinity>());
        info.dual_residual = std::max(info.dual_residual,
          rho_control * (slack[k] - old_slack).lpNorm<Eigen::Infinity>());
      }
    }
    info.iteration_count = iteration;
    if (!std::isfinite(info.primal_residual) || !std::isfinite(info.dual_residual)) return TinyMpcSolution{};
    if (info.primal_residual <= config.mpc_primal_tolerance
        && info.dual_residual <= config.mpc_dual_tolerance) {
      info.converged = true;
      break;
    }
    if (iteration % 10 == 0) {
      // 区域内自由移动需要较小角度惩罚，跨板空隙则需要较大惩罚累积乘子。
      // 两种分裂分别平衡残差，缩放对偶变量保持未缩放乘子不变。
      for (int axis = 0; axis < 2; ++axis) {
        double next = rho_angle[axis];
        if (angle_primal[axis] > 10.0 * angle_dual_residual[axis]) next *= 2.0;
        else if (angle_dual_residual[axis] > 10.0 * angle_primal[axis]) next *= 0.5;
        next = std::clamp(next, 1.0, std::max(1.0, weight[axis]));
        for (auto& dual : angle_dual) dual[axis] *= rho_angle[axis] / next;
        rho_angle[axis] = next;
      }
      double next = rho_control;
      if (control_primal > 10.0 * control_dual_residual) next *= 2.0;
      else if (control_dual_residual > 10.0 * control_primal) next *= 0.5;
      next = std::clamp(next, 1e-6, 1e6);
      for (auto& dual : control_dual) dual *= rho_control / next;
      rho_control = next;
      if (!precompute()) return TinyMpcSolution{};
    }
  }

  // 对投影后的加速度重新展开，严格满足加速度边界和离散动力学。
  for (int axis = 0; axis < 2; ++axis) {
    const auto& ref = axis == 0 ? reference.yaw : reference.pitch;
    auto& output = axis == 0 ? result.yaw : result.pitch;
    Eigen::Vector2d state{ref.angle.front(), ref.angular_velocity.front()};
    for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
      output.angle[k] = state.x(); output.angular_velocity[k] = state.y();
      output.acceleration[k] = slack[std::min(k, controls - 1)][axis];
      if (!state.allFinite()) return TinyMpcSolution{};
      if (k < controls) state = a * state + b * slack[k][axis];
    }
    output.info = info;
    output.valid = info.converged;
  }
  for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
    result.in_plate_offsets[k] = projectTinyMpcPlate(reference.regions[k],
      {result.yaw.angle[k], result.pitch.angle[k]}, weight);
  }
  result.valid = result.yaw.valid && result.pitch.valid;
  return result;
}

}  // namespace L4Planning
