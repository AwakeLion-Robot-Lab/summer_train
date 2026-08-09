#include "l3_estimation/ekf_.hpp"

#include <cmath>
#include <numeric>

namespace L3Estimation {
namespace {

// 卡方分布 95% 上分位数，下标即自由度。NIS 服从自由度等于观测维数的卡方
// 分布，所以门限必须随维数取值；写成固定常数会在观测维数变化时失配。
constexpr double kChiSquare95[] = {0.0,    3.841,  5.991,  7.815,
                                   9.488,  11.070, 12.592, 14.067,
                                   15.507, 16.919, 18.307, 19.675};
constexpr Eigen::Index kChiSquare95Count =
  static_cast<Eigen::Index>(sizeof(kChiSquare95) / sizeof(kChiSquare95[0]));

// 标准正态分布的 95% 分位数，用于超表自由度的 Wilson-Hilferty 近似。
constexpr double kNormal95 = 1.6448536269514722;

[[nodiscard]] double chiSquare95(Eigen::Index degrees_of_freedom) noexcept
{
  if (degrees_of_freedom >= 1 && degrees_of_freedom < kChiSquare95Count) {
    return kChiSquare95[degrees_of_freedom];
  }
  if (degrees_of_freedom < 1) {
    return 0.0;
  }

  // 超出查表范围时用 Wilson-Hilferty 近似，避免退化成某个固定门限。
  const double k = static_cast<double>(degrees_of_freedom);
  const double t =
    1.0 - 2.0 / (9.0 * k) + kNormal95 * std::sqrt(2.0 / (9.0 * k));
  return k * t * t * t;
}

}  // namespace

ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_minus)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add),
  x_minus(x_minus)
{
  // 预创建固定键，遥测侧可以在第一次更新前安全读取完整字段集合。
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  // F 传播协方差，f 负责实际状态转移，二者允许分别线性化和实现。
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  // H 是观测模型在当前状态处的 Jacobian。
  const Eigen::VectorXd x_prior = x;

  // 创新量和创新协方差必须在状态更新之前算好：更新本身就是在压缩残差，
  // 用后验 x 和后验 P 得到的量恒偏小，不再服从卡方分布。
  const Eigen::VectorXd residual = z_subtract(z, h(x));
  const Eigen::MatrixXd S = H * P * H.transpose() + R;
  const Eigen::MatrixXd S_inverse = S.inverse();

  Eigen::MatrixXd K = P * H.transpose() * S_inverse;

  // Joseph 形式能减小浮点误差对协方差对称性和半正定性的破坏。
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  // 通过 x_add 注入修正量，使角度等周期状态可以在加法后归一化。
  x = x_add(x, K * residual);

  if (consistency_mode == ConsistencyMode::SpPosterior) {
    // sp_vision 的算法：残差和 S 都取更新之后的值。更新本身就是在压缩残差，
    // 所以这个量恒偏小，不服从卡方分布——照抄是为了复现 sp 的复位行为，不是
    // 因为它对。见 SpCompatConfig::posterior_nis。
    const Eigen::VectorXd posterior_residual = z_subtract(z, h(x));
    const Eigen::MatrixXd posterior_S = H * P * H.transpose() + R;
    recordConsistency(posterior_residual, posterior_S.inverse(), x_prior);
    return x;
  }

  recordConsistency(residual, S_inverse, x_prior);
  return x;
}

void ExtendedKalmanFilter::recordConsistency(
  const Eigen::VectorXd & residual, const Eigen::MatrixXd & S_inverse,
  const Eigen::VectorXd & x_prior)
{
  // NIS 衡量观测与先验预测的一致性。这里的 nees 缺少真值，实际是本次状态
  // 修正量相对后验协方差的归一化幅度，只作为辅助的异常指标。
  double nis = residual.transpose() * S_inverse * residual;
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  // 门限取对应自由度的卡方 95% 上分位数：NIS 用观测维数，nees 用状态维数。
  //
  // sp_vision 复刻模式下两者都固定 0.711。那个数注释写的是"自由度 4、置信度
  // 95%"，但自由度 4 的卡方 95% **上**分位是 9.488，0.711 是**下** 5% 分位；
  // 配上同样错位的后验 NIS，sp 的 "Bad Converge" 复位是在拿一个偏小的统计量
  // 比一个偏小的门限。照抄是为了复现它的复位时机。
  const bool sp_mode = consistency_mode == ConsistencyMode::SpPosterior;
  constexpr double kSpFixedThreshold = 0.711;
  const double nis_threshold =
    sp_mode ? kSpFixedThreshold : chiSquare95(residual.rows());
  const double nees_threshold =
    sp_mode ? kSpFixedThreshold : chiSquare95(x.rows());

  // S 接近奇异时 nis 可能是 NaN，此时比较运算恒为假会被误判成通过，
  // 因此非有限值一律按失败计入。sp 没有这道保护，复刻模式下一并照抄：
  // NaN > 门限为假，会被当成"通过"，复位反而不会触发。
  const bool nis_fail =
    sp_mode ? (nis > nis_threshold) : (!std::isfinite(nis) || nis > nis_threshold);
  const bool nees_fail =
    sp_mode ? (nees > nees_threshold) : (!std::isfinite(nees) || nees > nees_threshold);

  if (nis_fail) nis_count_++;
  if (nees_fail) nees_count_++;
  // 逐次覆盖而不是只在失败时置 1，避免遥测里的标志位一直粘住。
  data["nis_fail"] = nis_fail ? 1.0 : 0.0;
  data["nees_fail"] = nees_fail ? 1.0 : 0.0;
  total_count_++;
  last_nis = nis;

  recent_nis_failures.push_back(nis_fail ? 1 : 0);

  // 只保留固定长度窗口，Tracker 使用该窗口判断滤波是否持续异常。
  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  // 当前四维观测顺序为 [方位角, 俯仰角, 距离, 装甲板 yaw]。观测维数不足
  // 时跳过，避免越界读取——本类不限定观测维数，调用方可能只更新子集。
  if (residual.rows() >= 4) {
    data["residual_yaw"] = residual[0];
    data["residual_pitch"] = residual[1];
    data["residual_distance"] = residual[2];
    data["residual_angle"] = residual[3];
  }
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;
}

}  // namespace L3Estimation
