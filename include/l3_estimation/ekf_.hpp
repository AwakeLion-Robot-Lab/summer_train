#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <deque>
#include <functional>
#include <map>
#include <string>

namespace L3Estimation {

// 支持自定义状态加法、非线性状态转移和非线性观测模型的扩展卡尔曼滤波器。
// 调用方负责保证各向量与矩阵维度一致。
class ExtendedKalmanFilter
{
public:
  // 当前后验状态和后验协方差。
  Eigen::VectorXd x;
  Eigen::MatrixXd P;

  ExtendedKalmanFilter() = default;

  // x_add 用于处理普通向量加法不适用的状态分量，例如周期角度归一化。
  ExtendedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; });

  // 使用线性状态转移 x = F*x 执行预测。
  Eigen::VectorXd predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q);

  // 使用调用方提供的非线性状态转移 f 执行预测；F 仍用于传播协方差。
  Eigen::VectorXd predict(
    const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

  // 使用线性观测模型 h(x) = H*x 执行更新。
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  // 使用调用方提供的非线性观测模型 h 执行更新；H 是该模型的 Jacobian。
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  // 最近一次残差与一致性统计量，供 Tracker 健康检查和遥测读取。
  std::map<std::string, double> data;
  // 滑动窗口中的 NIS 失败标记，1 表示该次更新超过门限。
  std::deque<int> recent_nis_failures{0};
  std::size_t window_size{100};
  double last_nis{0.0};

private:
  // 与状态维度相同的单位阵，以及用于注入修正量的状态加法函数。
  Eigen::MatrixXd I;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add{
    [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; }};

  int nees_count_ = 0;
  int nis_count_ = 0;
  int total_count_ = 0;
};


}  // namespace L3Estimation
