#include "l4_planning/mpc.hpp"

#include <algorithm>
#include <cmath>

namespace L4Planning {

namespace {

// Riccati 迭代的收敛判据与上限。TinyMPC 原本取 1e-5，实测在本模型上会给最终解
// 留下 7e-5 的相对偏差（对照稠密 QP 的精确解量的）；收紧到 1e-13 后降到 3.6e-13。
// 这一步只在 setup 跑一次，收紧不花任何运行时代价，没有理由留着那个误差。
constexpr double kRiccatiTolerance = 1e-13;
constexpr int kRiccatiMaxIterations = 1000;

bool positiveFinite(double value)
{
  return std::isfinite(value) && value > 0.0;
}

bool nonNegativeFinite(double value)
{
  return std::isfinite(value) && value >= 0.0;
}

}  // namespace

bool AxisMpc::setup(const Config& config)
{
  ready_ = false;

  // 视野至少要两步，否则没有一个输入可解。
  if (!positiveFinite(config.dt) || config.horizon < 2 ||
      config.max_iterations < 1 ||
      !nonNegativeFinite(config.q_position) ||
      !nonNegativeFinite(config.q_velocity) ||
      !nonNegativeFinite(config.r_input) ||
      !positiveFinite(config.rho) ||
      !positiveFinite(config.max_acceleration)) {
    return false;
  }
  // 位置权重为 0 时代价里根本没有"跟上参考"这一项，解会退化成原地不动。
  if (!(config.q_position > 0.0)) {
    return false;
  }

  config_ = config;
  const int steps = config_.horizon;

  A_ << 1.0, config_.dt, 0.0, 1.0;
  B_ << 0.0, config_.dt;

  // ADMM 把箱约束变成罚项，等价于在代价上加 rho·I，Riccati 用的是加过之后的
  // 权重——这一步和 TinyMPC 的 tiny_precompute_and_set_cache 一致。
  Q_aug_ << config_.q_position + config_.rho, config_.q_velocity + config_.rho;
  R_aug_ = config_.r_input + config_.rho;

  const Eigen::Matrix2d Q_diagonal = Q_aug_.asDiagonal();
  Eigen::Matrix2d P = config_.rho * Eigen::Matrix2d::Identity();
  Eigen::RowVector2d K = Eigen::RowVector2d::Zero();
  Eigen::RowVector2d K_previous = Eigen::RowVector2d::Zero();

  for (int iteration = 0; iteration < kRiccatiMaxIterations; ++iteration) {
    const double denominator = R_aug_ + B_.dot(P * B_);
    if (!positiveFinite(denominator)) {
      return false;
    }
    K = (B_.transpose() * P * A_) / denominator;
    P = Q_diagonal + A_.transpose() * P * (A_ - B_ * K);
    if ((K - K_previous).cwiseAbs().maxCoeff() < kRiccatiTolerance) {
      break;
    }
    K_previous = K;
  }

  const double Quu = R_aug_ + B_.dot(P * B_);
  if (!positiveFinite(Quu)) {
    return false;
  }

  K_inf_ = K;
  P_inf_ = P;
  Quu_inv_ = 1.0 / Quu;
  AmBKt_ = (A_ - B_ * K_inf_).transpose();

  x_.setZero(2, steps);
  p_.setZero(2, steps);
  q_.setZero(2, steps);
  v_.setZero(2, steps);
  g_.setZero(2, steps);
  u_.setZero(steps - 1);
  d_.setZero(steps - 1);
  r_.setZero(steps - 1);
  z_.setZero(steps - 1);
  y_.setZero(steps - 1);

  ready_ = true;
  return true;
}

void AxisMpc::reset() noexcept
{
  if (!ready_) {
    return;
  }
  x_.setZero();
  p_.setZero();
  q_.setZero();
  v_.setZero();
  g_.setZero();
  u_.setZero();
  d_.setZero();
  r_.setZero();
  z_.setZero();
  y_.setZero();
}

void AxisMpc::updateLinearCost(
  const Eigen::Ref<const Eigen::Matrix<double, 2, Eigen::Dynamic>>& x_ref)
{
  const int steps = config_.horizon;

  // 状态线性项：-Q·xref 再减去 ADMM 的罚项。
  //
  // **这里用原始 Q，不是 Q+rho**，和 TinyMPC 不同（它在 setup 里就把 work->Q 存成
  // Q+rho，update_linear_cost 直接拿它乘 Xref）。rho 那部分已经由下面的 -rho(v-g)
  // 承担：不动约束时 v 收敛到 x、g 收敛到 0，定点处梯度是 (Q+rho)x - Q·xref - rho·x
  // = Q(x - xref)，干净地以参考为中心。用 Q+rho 的话定点变成 Q·x - (Q+rho)·xref，
  // 相当于把参考放大 (1 + rho/Q) 倍。实测该偏置为 2.4e-4（q=1e3, rho=1），改用原始
  // Q 后降到 3.6e-13。sp 的 q=9e6 下这个偏置约 1e-7、可以忽略，但没有理由继承。
  q_.row(0) = -config_.q_position * x_ref.row(0);
  q_.row(1) = -config_.q_velocity * x_ref.row(1);
  q_.noalias() -= config_.rho * (v_ - g_);

  // 输入线性项。参考输入恒为 0（不预设加速度目标），所以只剩罚项。
  r_.noalias() = -config_.rho * (z_ - y_);

  // 终端项用 Pinf 代替 Q，等价于把视野之外的无穷段代价折进最后一步。
  p_.col(steps - 1).noalias() = -(P_inf_ * x_ref.col(steps - 1));
  p_.col(steps - 1).noalias() -= config_.rho * (v_.col(steps - 1) - g_.col(steps - 1));
}

bool AxisMpc::solve(
  const Eigen::Ref<const Eigen::Matrix<double, 2, Eigen::Dynamic>>& x_ref,
  const Eigen::Vector2d& x0)
{
  if (!ready_ || x_ref.cols() != config_.horizon || !x_ref.allFinite() ||
      !x0.allFinite()) {
    return false;
  }

  const int steps = config_.horizon;
  const double u_limit = config_.max_acceleration;

  x_.col(0) = x0;

  // 先按本帧的参考更新一次线性项再进循环。TinyMPC 的主循环是"先反向递推、最后
  // 更新线性项"，靠跨帧热启动把上一帧的线性项接上；但那样每次换了参考之后的第
  // 一次迭代用的还是上一帧的 xref，白跑一轮，冷启动时更是拿全零线性项开工。
  updateLinearCost(x_ref);

  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    // 反向递推：只更新线性项 d 和 p，二次项在 setup 时已经收敛成 Kinf/Pinf。
    for (int k = steps - 2; k >= 0; --k) {
      d_(k) = Quu_inv_ * (B_.dot(p_.col(k + 1)) + r_(k));
      p_.col(k).noalias() =
        q_.col(k) + AmBKt_ * p_.col(k + 1) - K_inf_.transpose() * r_(k);
    }

    // 正向滚动：用 LQR 反馈律把轨迹推出来。
    for (int k = 0; k < steps - 1; ++k) {
      u_(k) = -K_inf_.dot(x_.col(k)) - d_(k);
      x_.col(k + 1).noalias() = A_ * x_.col(k) + B_ * u_(k);
    }

    // 松弛变量投影到可行域。状态不设上下界（云台角度没有硬限位需要在这里管），
    // 所以状态那一路只是恒等投影，留着是为了和输入那一路对称、以后好加限位。
    const Eigen::MatrixXd v_new = x_ + g_;
    const Eigen::RowVectorXd z_new =
      (u_ + y_).cwiseMax(-u_limit).cwiseMin(u_limit);

    // 对偶更新。
    g_.noalias() += x_ - v_new;
    y_.noalias() += u_ - z_new;

    v_ = v_new;
    z_ = z_new;

    updateLinearCost(x_ref);
  }

  // 输出取松弛变量 z 而不是 u：z 一定落在 [-a_max, a_max] 里，u 在迭代没收敛时
  // 可能越界一点点。TinyMPC 的 solution->u 也是取 znew。
  u_ = z_;
  // 输入被夹过之后，状态要按夹过的输入重新滚一遍，否则 x 和 u 对不上，下游拿去
  // 算速度前馈就会自相矛盾。
  for (int k = 0; k < steps - 1; ++k) {
    x_.col(k + 1).noalias() = A_ * x_.col(k) + B_ * u_(k);
  }

  return true;
}

double AxisMpc::position(int step) const
{
  if (!ready_ || step < 0 || step >= config_.horizon) {
    return 0.0;
  }
  return x_(0, step);
}

double AxisMpc::velocity(int step) const
{
  if (!ready_ || step < 0 || step >= config_.horizon) {
    return 0.0;
  }
  return x_(1, step);
}

double AxisMpc::acceleration(int step) const
{
  if (!ready_ || step < 0 || step >= config_.horizon - 1) {
    return 0.0;
  }
  return u_(step);
}

}  // namespace L4Planning
