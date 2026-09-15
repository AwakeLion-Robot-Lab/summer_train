#pragma once

#include "l3_estimation/ekf_.hpp"

#include <Eigen/Dense>

#include <functional>

namespace L3Estimation {

// 迭代扩展卡尔曼滤波器 (IEKF / IESKF)。
//
// 普通 EKF 只在先验点线性化一次。当观测模型强非线性时——本项目里就是
// h_armor_xyz 之后再叠一层 xyz2ypd 球坐标映射——先验点离真实后验越远，
// Jacobian 的失真越大，收敛越慢甚至偏到错误的不动点。整车模型中 r1、
// r2-r1、z2-z1 这三个分量观测最弱，受线性化误差影响也最明显。
//
// 本类用 Gauss-Newton 反复把线性化工作点拉向后验，对如下 MAP 代价求极小：
//
//   J(x) = ||x ⊟ x_pri||²_{P_pri⁻¹} + ||z - h(x)||²_{R⁻¹}
//
// 迭代式 (Bell & Cathey, 1993)：
//
//   x_{i+1} = x_pri ⊞ K_i [ z - h(x_i) - H_i (x_pri ⊟ x_i) ]
//   K_i     = P_pri H_iᵀ (H_i P_pri H_iᵀ + R)⁻¹
//
// 注意末项是**减去** H_i (x_pri ⊟ x_i)。SHtech_auto_aim 的 IESEKF.hpp 里
// 写成了加号：第 0 次迭代 dx_pri 为零看不出差别，从第二次起会收敛到一个
// 有偏的不动点。移植时务必核对这个符号。
//
// 设计约束：
//  - 每次迭代都要重算 Jacobian，所以 H 是函数而不是矩阵。
//  - K 和协方差传播始终以先验为锚点，迭代只移动工作点，不会把同一次观测
//    重复吸收进状态。
//  - NIS 取先验线性化点的创新量。迭代后的残差被压缩过，不再服从自由度等于
//    观测维数的卡方分布，直接拿来记会让 Tracker::badRecentNis 的 40% 失败率
//    门限失配、复位保护变哑。
//  - max_iterations 为 1 时与基类 update() 逐位等价，便于离线回放做 A/B。
class IteratedKalmanFilter : public ExtendedKalmanFilter
{
public:
  using ExtendedKalmanFilter::ExtendedKalmanFilter;
  using ExtendedKalmanFilter::update;

  IteratedKalmanFilter() = default;

  // H_of 返回观测模型在给定状态处的 Jacobian，逐次迭代重新求值。
  // z_subtract 负责观测空间的流形减法（角度分量必须走最短圆周差）。
  Eigen::VectorXd update(
    const Eigen::VectorXd & z,
    const std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> & H_of,
    const Eigen::MatrixXd & R,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> & h,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> &
      z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; },
    int max_iterations = 5,
    double step_threshold = 1e-4);

  // 最近一次更新实际执行的迭代次数，供遥测判断是否触及上限。
  [[nodiscard]] int lastIterations() const noexcept { return last_iterations_; }

private:
  int last_iterations_{0};
};

}  // namespace L3Estimation
