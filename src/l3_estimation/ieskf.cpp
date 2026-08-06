#include "l3_estimation/ieskf.hpp"

namespace L3Estimation {

Eigen::VectorXd IteratedKalmanFilter::update(
  const Eigen::VectorXd & z,
  const std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> & H_of,
  const Eigen::MatrixXd & R,
  const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> & h,
  const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> &
    z_subtract,
  int max_iterations,
  double step_threshold)
{
  // 迭代全程以先验为锚点，成员 x、P 直到最后才写回。
  const Eigen::VectorXd x_prior = x;
  const Eigen::MatrixXd P_prior = P;

  // 第 0 次迭代的工作点就是先验，此时 dx_pri 为零，结果与非迭代 update()
  // 完全一致；它同时充当后续迭代发散时的回退点。
  Eigen::MatrixXd H = H_of(x_prior);
  const Eigen::VectorXd residual_prior = z_subtract(z, h(x_prior));
  const Eigen::MatrixXd S_prior = H * P_prior * H.transpose() + R;
  const Eigen::MatrixXd S_prior_inverse = S_prior.inverse();
  Eigen::MatrixXd K = P_prior * H.transpose() * S_prior_inverse;

  // 增益非有限时整条更新都不可信，直接退回先验，不改动状态和协方差。
  if (!K.allFinite() || !residual_prior.allFinite()) {
    recordConsistency(residual_prior, S_prior_inverse, x_prior);
    last_iterations_ = 0;
    return x;
  }

  Eigen::VectorXd x_iterate = x_add(x_prior, K * residual_prior);
  if (!x_iterate.allFinite()) {
    x_iterate = x_prior;
  }

  // 协方差用最后一次成功迭代的 H、K 传播，所以要跟着工作点一起保存。
  Eigen::MatrixXd H_final = H;
  Eigen::MatrixXd K_final = K;
  last_iterations_ = 1;

  for (int iteration = 1; iteration < max_iterations; ++iteration) {
    H = H_of(x_iterate);
    const Eigen::MatrixXd S = H * P_prior * H.transpose() + R;
    K = P_prior * H.transpose() * S.inverse();

    const Eigen::VectorXd innovation = z_subtract(z, h(x_iterate));
    // dx_prior = x_pri ⊟ x_i，必须走 x_minus 才能正确处理 yaw 的周期性。
    const Eigen::VectorXd dx_prior = x_minus(x_prior, x_iterate);
    const Eigen::VectorXd x_next =
      x_add(x_prior, K * (innovation - H * dx_prior));

    // 任何一步出现非有限值就停在上一轮的可用结果上，绝不写回滤波器。
    if (!K.allFinite() || !x_next.allFinite()) {
      break;
    }

    // 收敛判据是相邻两次工作点之差。11 维状态混了米、米每秒和弧度，这个
    // 范数偏保守：达不到阈值只会多跑几次迭代，被 max_iterations 兜住。
    const double step = x_minus(x_next, x_iterate).norm();
    x_iterate = x_next;
    H_final = H;
    K_final = K;
    last_iterations_ = iteration + 1;

    if (step < step_threshold) {
      break;
    }
  }

  // Joseph 形式，且只从先验传播一次——迭代不重复吸收观测。
  const Eigen::MatrixXd IKH = I - K_final * H_final;
  P = IKH * P_prior * IKH.transpose() + K_final * R * K_final.transpose();
  x = x_iterate;

  // 一致性统计固定用先验线性化点的创新量，与非迭代路径口径一致。
  recordConsistency(residual_prior, S_prior_inverse, x_prior);
  return x;
}

}  // namespace L3Estimation
