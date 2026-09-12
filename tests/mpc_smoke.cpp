// 单轴 MPC 的行为冒烟测试。只依赖 Eigen。
//
// 验证策略：不去比对另一份 ADMM 实现（同样的思路写两遍，共有的错会一起躲过），
// 而是把同一个二次规划**稠密地**写出来——把状态显式消成输入的线性函数，得到
// H、c，然后：
//   1. 约束不起作用时，最优解就是线性方程 H u = -c 的解，可以精确求出来比对；
//   2. 约束起作用时，检查 KKT 条件（内点处梯度为零、贴边处梯度指向边界外）。
// 这两条都只依赖"问题是什么"，不依赖"用什么算法解"，所以能真正证伪实现。

#include "l4_planning/mpc.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "mpc smoke test failed: " << message << '\n';
    std::exit(1);
  }
}

L4Planning::AxisMpc::Config baseConfig()
{
  L4Planning::AxisMpc::Config config;
  config.dt = 0.01;
  config.horizon = 40;
  config.q_position = 1.0e3;
  config.q_velocity = 0.0;
  config.r_input = 1.0;
  config.rho = 1.0;
  config.max_iterations = 4000;
  config.max_acceleration = 1.0e9;  // 大到约束一定不起作用
  return config;
}

// 参考轨迹：前 15 步停在 offset，之后阶跃回 0 并保持到末尾。切板在射击轨迹上
// 就是这个形状。末段必须回到 0——终端代价用的是 Riccati 的 Pinf，只有参考末值
// 为零时它才和"以参考为中心的二次型"完全一致，稠密比对才有意义。
Eigen::Matrix<double, 2, Eigen::Dynamic> stepReference(
  int horizon, int step_index, double offset)
{
  Eigen::Matrix<double, 2, Eigen::Dynamic> reference(2, horizon);
  reference.setZero();
  for (int k = 0; k < step_index && k < horizon; ++k) {
    reference(0, k) = offset;
  }
  return reference;
}

// 把 x = S·u + T·x0 显式写出来，再拼出稠密 QP 的 H 和 c。
// 代价 J(u) = (S u + T x0 − Xref)ᵀ W (S u + T x0 − Xref) + R·uᵀu，
// 其中 W 是分块对角：前 horizon−1 块是 Q，最后一块是终端权重 P。
struct DenseProblem {
  Eigen::MatrixXd H;
  Eigen::VectorXd c;
  Eigen::MatrixXd S;
  Eigen::MatrixXd T;
  Eigen::VectorXd W_diagonal_position;  // 仅用于组装，保留以便调试
};

// 终端权重：按和实现一致的方式，对 Q+rho 做 Riccati 迭代得到 Pinf。
Eigen::Matrix2d terminalWeight(const L4Planning::AxisMpc::Config & config)
{
  Eigen::Matrix2d A;
  A << 1.0, config.dt, 0.0, 1.0;
  Eigen::Vector2d B;
  B << 0.0, config.dt;

  Eigen::Vector2d q_augmented;
  q_augmented << config.q_position + config.rho, config.q_velocity + config.rho;
  const Eigen::Matrix2d Q = q_augmented.asDiagonal();
  const double R = config.r_input + config.rho;

  Eigen::Matrix2d P = config.rho * Eigen::Matrix2d::Identity();
  Eigen::RowVector2d K_previous = Eigen::RowVector2d::Zero();
  for (int i = 0; i < 1000; ++i) {
    const Eigen::RowVector2d K = (B.transpose() * P * A) / (R + B.dot(P * B));
    P = Q + A.transpose() * P * (A - B * K);
    if ((K - K_previous).cwiseAbs().maxCoeff() < 1e-14) {
      break;
    }
    K_previous = K;
  }
  return P;
}

DenseProblem buildDenseProblem(
  const L4Planning::AxisMpc::Config & config,
  const Eigen::Matrix<double, 2, Eigen::Dynamic> & reference,
  const Eigen::Vector2d & x0)
{
  const int horizon = config.horizon;
  const int inputs = horizon - 1;

  Eigen::Matrix2d A;
  A << 1.0, config.dt, 0.0, 1.0;
  Eigen::Vector2d B;
  B << 0.0, config.dt;

  DenseProblem problem;
  problem.S = Eigen::MatrixXd::Zero(2 * horizon, inputs);
  problem.T = Eigen::MatrixXd::Zero(2 * horizon, 2);

  Eigen::Matrix2d A_power = Eigen::Matrix2d::Identity();
  problem.T.block<2, 2>(0, 0) = A_power;
  for (int k = 1; k < horizon; ++k) {
    A_power = A * A_power;
    problem.T.block<2, 2>(2 * k, 0) = A_power;
    for (int j = 0; j < k; ++j) {
      // x_k 里 u_j 的系数是 A^(k-1-j) B。
      Eigen::Matrix2d A_step = Eigen::Matrix2d::Identity();
      for (int p = 0; p < k - 1 - j; ++p) {
        A_step = A * A_step;
      }
      problem.S.block<2, 1>(2 * k, j) = A_step * B;
    }
  }

  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(2 * horizon, 2 * horizon);
  for (int k = 0; k < horizon - 1; ++k) {
    W(2 * k, 2 * k) = config.q_position;
    W(2 * k + 1, 2 * k + 1) = config.q_velocity;
  }
  // 终端权重是 Pinf 减去 rho·I：实现里终端线性项取 -Pinf·xref - rho·x，与终端二次
  // 项 Pinf 合起来，梯度是 (Pinf - rho·I)x - Pinf·xref。参考末值为 0 时后一项消失，
  // 等效权重就是 Pinf - rho·I。不减这个 rho 会留下 7e-5 量级的假失配。
  W.block<2, 2>(2 * (horizon - 1), 2 * (horizon - 1)) =
    terminalWeight(config) - config.rho * Eigen::Matrix2d::Identity();

  Eigen::VectorXd reference_stacked(2 * horizon);
  for (int k = 0; k < horizon; ++k) {
    reference_stacked(2 * k) = reference(0, k);
    reference_stacked(2 * k + 1) = reference(1, k);
  }

  const Eigen::VectorXd bias = problem.T * x0 - reference_stacked;
  problem.H = 2.0 * (problem.S.transpose() * W * problem.S +
                     config.r_input * Eigen::MatrixXd::Identity(inputs, inputs));
  problem.c = 2.0 * (problem.S.transpose() * W * bias);
  return problem;
}

Eigen::VectorXd solvedInputs(const L4Planning::AxisMpc & mpc, int horizon)
{
  Eigen::VectorXd u(horizon - 1);
  for (int k = 0; k < horizon - 1; ++k) {
    u(k) = mpc.acceleration(k);
  }
  return u;
}

// ---- 1. 参数校验：非法配置必须拒绝，而不是悄悄用一个凑合的值 ----
void testSetupRejectsBadConfig()
{
  L4Planning::AxisMpc mpc;

  auto bad = [&](L4Planning::AxisMpc::Config config, const char * what) {
    L4Planning::AxisMpc local;
    require(!local.setup(config), what);
    require(!local.ready(), "rejected config must leave the solver unusable");
  };

  auto config = baseConfig();
  config.dt = 0.0;
  bad(config, "dt = 0 must be rejected");

  config = baseConfig();
  config.horizon = 1;
  bad(config, "horizon < 2 must be rejected");

  config = baseConfig();
  config.q_position = 0.0;
  bad(config, "zero position weight must be rejected");

  config = baseConfig();
  config.max_acceleration = 0.0;
  bad(config, "zero acceleration limit must be rejected");

  config = baseConfig();
  config.rho = 0.0;
  bad(config, "zero rho must be rejected");

  require(mpc.setup(baseConfig()), "a sane config must be accepted");
  require(mpc.ready(), "ready() must be true after a successful setup");

  // 没 setup 过的实例不能假装解出了东西。
  L4Planning::AxisMpc untouched;
  Eigen::Matrix<double, 2, Eigen::Dynamic> reference(2, 40);
  reference.setZero();
  require(
    !untouched.solve(reference, Eigen::Vector2d::Zero()),
    "solve() on an un-setup solver must fail");

  std::cout << "  [ok] bad configs are rejected, solver stays unusable\n";
}

// ---- 2. 约束不起作用时，解必须等于稠密 QP 的精确解 ----
void testMatchesDenseOptimumWhenUnconstrained()
{
  const auto config = baseConfig();
  const auto reference = stepReference(config.horizon, 15, 0.10);
  const Eigen::Vector2d x0 = reference.col(0);

  L4Planning::AxisMpc mpc;
  require(mpc.setup(config), "setup must succeed");
  require(mpc.solve(reference, x0), "solve must succeed");

  const auto problem = buildDenseProblem(config, reference, x0);
  const Eigen::VectorXd exact = problem.H.ldlt().solve(-problem.c);
  const Eigen::VectorXd actual = solvedInputs(mpc, config.horizon);

  const double error = (exact - actual).cwiseAbs().maxCoeff();
  const double scale = exact.cwiseAbs().maxCoeff();
  require(scale > 1.0, "the test case must actually excite the solver");
  // 4000 次迭代下实测 3.6e-13；留三个数量级余量，真出偏差一定抓得住。
  require(
    error / scale < 1e-9,
    "unconstrained solution must match the dense QP optimum");

  std::cout << "  [ok] unconstrained solution matches dense optimum, rel err "
            << error / scale << '\n';
}

// ---- 3. 约束起作用时：可行 + 满足 KKT ----
void testConstrainedSolutionIsFeasibleAndOptimal()
{
  auto config = baseConfig();
  // 这组权重下无约束解的峰值加速度约 2.5 rad/s^2，上限取 1.0 才真的夹得住。
  // 夹不住的话这个用例会退化成上一个，什么也证明不了，所以下面还要断言确实夹到了。
  config.max_acceleration = 1.0;
  const auto reference = stepReference(config.horizon, 15, 0.10);
  const Eigen::Vector2d x0 = reference.col(0);

  L4Planning::AxisMpc mpc;
  require(mpc.setup(config), "setup must succeed");
  require(mpc.solve(reference, x0), "solve must succeed");

  const Eigen::VectorXd u = solvedInputs(mpc, config.horizon);

  // 可行性：这是给云台的硬承诺，一步都不能越界。
  require(
    u.cwiseAbs().maxCoeff() <= config.max_acceleration + 1e-9,
    "no input may exceed the acceleration limit");

  // 约束必须真的起作用，否则这个用例退化成上一个。
  int saturated = 0;
  for (int k = 0; k < u.size(); ++k) {
    if (std::abs(std::abs(u(k)) - config.max_acceleration) < 1e-6) {
      ++saturated;
    }
  }
  require(saturated > 0, "the limit must actually bind in this case");

  // KKT：内点处梯度为零；贴上界时梯度 <= 0，贴下界时梯度 >= 0。
  const auto problem = buildDenseProblem(config, reference, x0);
  const Eigen::VectorXd gradient = problem.H * u + problem.c;
  const double gradient_scale = problem.c.cwiseAbs().maxCoeff();
  const double tolerance = 1e-3 * gradient_scale;

  for (int k = 0; k < u.size(); ++k) {
    if (u(k) > config.max_acceleration - 1e-6) {
      require(gradient(k) <= tolerance, "gradient sign wrong at upper bound");
    } else if (u(k) < -config.max_acceleration + 1e-6) {
      require(gradient(k) >= -tolerance, "gradient sign wrong at lower bound");
    } else {
      require(
        std::abs(gradient(k)) <= tolerance,
        "gradient must vanish in the interior");
    }
  }

  std::cout << "  [ok] constrained solution feasible and KKT-optimal, "
            << saturated << "/" << u.size() << " steps saturated\n";
}

// ---- 4. 动力学自洽：输出的位置/速度必须和输出的加速度对得上 ----
void testStatesAreConsistentWithInputs()
{
  auto config = baseConfig();
  config.max_acceleration = 1.0;  // 夹得住，这样也覆盖到"被夹过之后重新滚状态"
  config.max_iterations = 10;  // 按实际部署的迭代次数
  const auto reference = stepReference(config.horizon, 15, 0.10);
  const Eigen::Vector2d x0 = reference.col(0);

  L4Planning::AxisMpc mpc;
  require(mpc.setup(config), "setup must succeed");
  require(mpc.solve(reference, x0), "solve must succeed");

  double worst = 0.0;
  for (int k = 0; k + 1 < config.horizon; ++k) {
    const double predicted_position =
      mpc.position(k) + config.dt * mpc.velocity(k);
    const double predicted_velocity =
      mpc.velocity(k) + config.dt * mpc.acceleration(k);
    worst = std::max(worst, std::abs(mpc.position(k + 1) - predicted_position));
    worst = std::max(worst, std::abs(mpc.velocity(k + 1) - predicted_velocity));
  }
  require(worst < 1e-9, "states must follow x[k+1] = A x[k] + B u[k]");

  // 迭代次数砍到 10 次（实际部署值）后，可行性仍然必须成立。
  for (int k = 0; k + 1 < config.horizon; ++k) {
    require(
      std::abs(mpc.acceleration(k)) <= config.max_acceleration + 1e-9,
      "10-iteration solution must still respect the limit");
  }

  std::cout << "  [ok] states consistent with inputs, worst drift " << worst
            << '\n';
}

// ---- 5. 提前减速：台阶还没到，云台就该动了 ----
// 这是整个方案存在的理由。参考轨迹在第 20 步才跳，如果解出来的加速度在第 20 步
// 之前一直是 0，那就退化成了"跟着阶跃硬追"，和不做规划没有区别。
void testBrakesBeforeTheStep()
{
  auto config = baseConfig();
  config.q_position = 9.0e6;  // 实际部署的权重
  config.r_input = 1.0;
  config.max_acceleration = 40.0;
  config.max_iterations = 10;
  config.horizon = 60;

  const int step_index = 20;
  const auto reference = stepReference(config.horizon, step_index, 0.15);
  const Eigen::Vector2d x0 = reference.col(0);

  L4Planning::AxisMpc mpc;
  require(mpc.setup(config), "setup must succeed");
  require(mpc.solve(reference, x0), "solve must succeed");

  double peak_before = 0.0;
  for (int k = 0; k < step_index; ++k) {
    peak_before = std::max(peak_before, std::abs(mpc.acceleration(k)));
  }
  require(
    peak_before > 0.1 * config.max_acceleration,
    "the solver must start moving before the step, not after it");

  // 台阶之后要真的走到位：末段参考是 0，位置应当收敛到 0 附近。
  const double terminal_error = std::abs(mpc.position(config.horizon - 1));
  require(
    terminal_error < 0.02 * 0.15,
    "the solution must actually reach the new reference by the horizon end");

  // 全程不越界。
  for (int k = 0; k + 1 < config.horizon; ++k) {
    require(
      std::abs(mpc.acceleration(k)) <= config.max_acceleration + 1e-9,
      "pre-braking must stay inside the acceleration limit");
  }

  std::cout << "  [ok] brakes before the step, peak |acc| before step "
            << peak_before << " rad/s^2, terminal error " << terminal_error
            << " rad\n";
}

// ---- 6. reset() 必须真的清掉热启动 ----
// 求解器跨帧复用上一次的解和对偶变量。目标丢失后如果不清，新目标的第一帧会带着
// 旧目标的偏置迭代；这个测试保证 reset() 之后的结果只由本次入参决定。
void testResetClearsWarmStart()
{
  auto config = baseConfig();
  config.max_acceleration = 40.0;
  config.max_iterations = 10;

  const auto reference = stepReference(config.horizon, 15, 0.10);
  const Eigen::Vector2d x0 = reference.col(0);

  L4Planning::AxisMpc fresh;
  require(fresh.setup(config), "setup must succeed");
  require(fresh.solve(reference, x0), "solve must succeed");
  const Eigen::VectorXd expected = solvedInputs(fresh, config.horizon);

  L4Planning::AxisMpc polluted;
  require(polluted.setup(config), "setup must succeed");
  // 先用一条完全不同的参考污染热启动状态。
  const auto other = stepReference(config.horizon, 30, -0.30);
  require(polluted.solve(other, other.col(0)), "solve must succeed");
  const Eigen::VectorXd polluted_first = solvedInputs(polluted, config.horizon);

  polluted.reset();
  require(polluted.solve(reference, x0), "solve must succeed");
  const Eigen::VectorXd after_reset = solvedInputs(polluted, config.horizon);

  require(
    (polluted_first - expected).cwiseAbs().maxCoeff() > 1.0,
    "the polluting case must really differ, otherwise this proves nothing");
  require(
    (after_reset - expected).cwiseAbs().maxCoeff() < 1e-9,
    "after reset() the result must depend only on this call's inputs");

  std::cout << "  [ok] reset() clears the warm start\n";
}

}  // namespace

int main()
{
  testSetupRejectsBadConfig();
  testMatchesDenseOptimumWhenUnconstrained();
  testConstrainedSolutionIsFeasibleAndOptimal();
  testStatesAreConsistentWithInputs();
  testBrakesBeforeTheStep();
  testResetClearsWarmStart();
  std::cout << "mpc smoke test passed\n";
  return 0;
}
