#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <deque>
#include <functional>
#include <map>
#include <string>

namespace L3Estimation::FilterEst {

// 扩展卡尔曼滤波器。状态加法、状态转移和观测模型都可由调用方替换，因为整车
// 模型的 yaw 是周期量，普通向量加减会在 ±pi 处跳变。
class ExtendedKalmanFilter
{
public:
  Eigen::VectorXd x;  // 后验状态
  Eigen::MatrixXd P;  // 后验协方差

  ExtendedKalmanFilter() = default;

  // x_add 负责把修正量注入状态，角度分量在这里归一化。
  ExtendedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; });

  Eigen::VectorXd predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q);

  // 非线性状态转移 f；F 仍然用于传播协方差。
  Eigen::VectorXd predict(
    const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  // 非线性观测模型 h，H 是它的 Jacobian；z_subtract 处理角度残差。
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  // 最近一次残差与一致性统计量，供遥测读取。
  std::map<std::string, double> data;
  // 滑动窗口内的 NIS 失败标记，Tracker 用它判断滤波是否持续异常。
  std::deque<int> recent_nis_failures{0};
  std::size_t window_size{100};
  double last_nis{0.0};

private:
  Eigen::MatrixXd I;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add{
    [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; }};
};

}  // namespace L3Estimation::FilterEst
