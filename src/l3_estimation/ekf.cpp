#include "l3_estimation/ekf.hpp"

#include <numeric>

namespace L3Estimation {

ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
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
  Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

  // Joseph 形式能减小浮点误差对协方差对称性和半正定性的破坏。
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  // 通过 x_add 注入修正量，使角度等周期状态可以在加法后归一化。
  x = x_add(x, K * z_subtract(z, h(x)));

  // 卡方检验。残差和 S 都取**更新之后**的值，门限固定 0.711——这是
  // sp_vision 的算法，逐行照抄。两点都与教科书不符，改动前先读完：
  //  - 更新本身就是在压缩残差，后验残差配后验 S 得到的量恒偏小，不再服从
  //    自由度等于观测维数的卡方分布；
  //  - 0.711 是自由度 4 的卡方**下** 5% 分位，95% 上分位是 9.488。
  // 两处偏差方向相反，Tracker::track 的 "Bad Converge" 复位时机就建立在这
  // 组数值上，单独"修好"任何一项都会改变复位频率。
  Eigen::VectorXd residual = z_subtract(z, h(x));
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double nis = residual.transpose() * S.inverse() * residual;
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  constexpr double nis_threshold = 0.711;
  constexpr double nees_threshold = 0.711;

  if (nis > nis_threshold) nis_count_++, data["nis_fail"] = 1;
  if (nees > nees_threshold) nees_count_++, data["nees_fail"] = 1;
  total_count_++;
  last_nis = nis;

  recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);

  // 只保留固定长度窗口，Tracker 使用该窗口判断滤波是否持续异常。
  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  // 当前四维观测顺序为 [方位角, 俯仰角, 距离, 装甲板 yaw]。观测维数不足时
  // 跳过，避免越界读取——本类不限定观测维数，调用方可能只更新子集。
  if (residual.rows() >= 4) {
    data["residual_yaw"] = residual[0];
    data["residual_pitch"] = residual[1];
    data["residual_distance"] = residual[2];
    data["residual_angle"] = residual[3];
  }
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;

  return x;
}

}  // namespace L3Estimation
