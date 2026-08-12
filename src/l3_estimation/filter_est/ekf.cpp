#include "l3_estimation/filter_est/ekf.hpp"

#include <numeric>
#include <utility>

namespace L3Estimation::FilterEst {

ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(std::move(x_add))
{
  // 预创建固定键，遥测侧在第一次更新前也能读到完整字段集合。
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

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
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
  const Eigen::VectorXd x_prior = x;
  const Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

  // Joseph 形式：比 (I - KH)P 更能保住协方差的对称性和半正定性。
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  x = x_add(x, K * z_subtract(z, h(x)));

  // 卡方检验。残差和 S 都取更新**之后**的值，门限 0.711 是与之配套的经验值，
  // Tracker 的 "Bad Converge" 复位时机建立在这组数值上，单独改任何一项都会
  // 改变复位频率。
  const Eigen::VectorXd residual = z_subtract(z, h(x));
  const Eigen::MatrixXd S = H * P * H.transpose() + R;
  const double nis = residual.transpose() * S.inverse() * residual;
  const double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  constexpr double kNisThreshold = 0.711;
  constexpr double kNeesThreshold = 0.711;

  last_nis = nis;
  recent_nis_failures.push_back(nis > kNisThreshold ? 1 : 0);
  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  const int recent_failures =
    std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);

  data["residual_yaw"] = residual[0];
  data["residual_pitch"] = residual[1];
  data["residual_distance"] = residual[2];
  data["residual_angle"] = residual[3];
  data["nis"] = nis;
  data["nees"] = nees;
  data["nis_fail"] = nis > kNisThreshold ? 1.0 : 0.0;
  data["nees_fail"] = nees > kNeesThreshold ? 1.0 : 0.0;
  data["recent_nis_failures"] =
    static_cast<double>(recent_failures) / static_cast<double>(recent_nis_failures.size());

  return x;
}

}  // namespace L3Estimation::FilterEst
