#pragma once

#include <Eigen/Core>

#include <vector>

namespace L4Planning {

// 单轴轨迹规划器：把一条"应该指哪里"的参考轨迹，投影到云台加速度能力允许的
// 范围内。与 sp_vision 2025 readme 4.4 的"方案一·隐式搜索"同构：
//
//   状态 x = [角度, 角速度]   输入 u = 角加速度
//   x[k+1] = A x[k] + B u[k]   A = [[1, dt], [0, 1]]   B = [0, dt]
//
//   min  Σ (x[k] - xref[k])ᵀ Q (x[k] - xref[k]) + Σ R u[k]²
//   s.t. |u[k]| <= max_acceleration
//
// Q 只罚位置、不罚速度（q_velocity 默认 0），代价就是 readme 里的"重合度"：
// 规划轨迹尽量贴着射击轨迹走，只在加速度不够时才偏离。切板在参考轨迹里表现为
// 一个台阶，求解器会在台阶之前就开始转，这就是"提前减速"。
//
// **它不是闭环控制器**：x[0] 取参考轨迹的首点，而不是云台实测姿态——sp 的
// readme 原话是"其输入不包含云台的实际状态"。闭环留给下位机，上位机只负责给出
// 一条它追得动的参考。sp 也试过把 MPC 当闭环控制器直接下发力矩，实测更差。
//
// 求解用 TinyMPC 的 ADMM 递推（Nguyen 等，ICRA 2024）：无限时域 LQR 增益在
// setup 时一次算好，此后每次迭代只有反向递推、正向滚动、箱投影、对偶更新四步，
// 不做任何矩阵分解。这里按 nx=2 / nu=1 专门化，省掉通用实现的动态分配。
class AxisMpc
{
public:
  struct Config {
    double dt{0.01};                // 离散步长，秒
    int horizon{100};               // 步数；dt * horizon 就是整个视野长度
    double q_position{9.0e6};       // 位置偏差权重
    double q_velocity{0.0};         // 速度偏差权重
    double r_input{1.0};            // 加速度代价权重
    double rho{1.0};                // ADMM 罚参数
    int max_iterations{10};         // 每次求解的迭代上限
    double max_acceleration{50.0};  // |u| 上限，rad/s^2
  };

  // 预计算 Riccati 缓存。参数非法（dt/horizon/权重/上限不合理）时返回 false，
  // 此后 ready() 一直是 false，solve() 直接拒绝——不造假、不退化成无约束解。
  bool setup(const Config& config);
  [[nodiscard]] bool ready() const noexcept { return ready_; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }

  // 丢弃热启动状态。求解器跨帧复用上一次的解和对偶变量，10 次迭代才够用；
  // 目标丢失或参考轨迹不再连续时必须调一次，否则会带着旧解的偏置继续迭代。
  void reset() noexcept;

  // x_ref 必须是 2 x horizon：第 0 行角度、第 1 行角速度，单位 rad 与 rad/s。
  // x0 是轨迹起点的状态，取 x_ref 的首列即可（见类注释）。
  // 返回 false 表示没 setup 过或入参尺寸不对；迭代到上限不算失败。
  bool solve(const Eigen::Ref<const Eigen::Matrix<double, 2, Eigen::Dynamic>>& x_ref,
             const Eigen::Vector2d& x0);

  // 求解结果。索引范围 [0, horizon)，加速度只到 horizon - 1。
  [[nodiscard]] double position(int step) const;
  [[nodiscard]] double velocity(int step) const;
  [[nodiscard]] double acceleration(int step) const;

private:
  void updateLinearCost(
    const Eigen::Ref<const Eigen::Matrix<double, 2, Eigen::Dynamic>>& x_ref);

  Config config_{};
  bool ready_{false};

  // 离散模型与 Riccati 缓存。
  Eigen::Matrix2d A_{Eigen::Matrix2d::Identity()};
  Eigen::Vector2d B_{Eigen::Vector2d::Zero()};
  Eigen::RowVector2d K_inf_{Eigen::RowVector2d::Zero()};
  Eigen::Matrix2d P_inf_{Eigen::Matrix2d::Zero()};
  Eigen::Matrix2d AmBKt_{Eigen::Matrix2d::Zero()};
  double Quu_inv_{0.0};
  Eigen::Vector2d Q_aug_{Eigen::Vector2d::Zero()};  // Q + rho，对角
  double R_aug_{0.0};                               // R + rho

  // 工作区。跨帧保留即为热启动。
  Eigen::Matrix<double, 2, Eigen::Dynamic> x_;
  Eigen::Matrix<double, 2, Eigen::Dynamic> p_;
  Eigen::Matrix<double, 2, Eigen::Dynamic> q_;
  Eigen::Matrix<double, 2, Eigen::Dynamic> v_;  // 状态松弛
  Eigen::Matrix<double, 2, Eigen::Dynamic> g_;  // 状态对偶
  Eigen::RowVectorXd u_;
  Eigen::RowVectorXd d_;
  Eigen::RowVectorXd r_;
  Eigen::RowVectorXd z_;  // 输入松弛
  Eigen::RowVectorXd y_;  // 输入对偶
};

}  // namespace L4Planning
