#pragma once

#include <ceres/jet.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

// 误差状态扩展卡尔曼滤波器，状态维数和状态转移由模板参数给定。
//
// 与普通 EKF 的区别只有一句话：协方差 P 描述的是误差状态 δ，不是状态 x。状态
// 拆成流形上的名义状态和切空间里的误差状态，卡尔曼增益算出的是 δ，再由 ⊞ 注入
// 回名义状态。δ 恒在零附近，旋转碰不到奇异点，而切空间是真正的向量空间，线性
// 代数都合法。
//
// H 用中心差分求，不用 Jet：ObsBase 是类型擦除的虚接口，只认 VectorXd，Jet 流
// 不过去。逐行拆解见 docs/esekf_uvl_port.md 第 6 节。
namespace L3Estimation {

template <int N_X, class PredictFunc>
class ErrorStateEkf
{
public:
  using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
  using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
  using Jet = ceres::Jet<double, N_X>;
  using JetMatrixX1 = Eigen::Matrix<Jet, N_X, 1>;

  using UpdateQFunc = std::function<MatrixXX()>;
  using InjectFunc = std::function<void(const MatrixX1 &, MatrixX1 &)>;
  using BoxMinusFunc = std::function<void(const MatrixX1 &, const MatrixX1 &, MatrixX1 &)>;
  using InjectJetFunc = std::function<void(const JetMatrixX1 &, JetMatrixX1 &)>;
  using BoxMinusJetFunc =
    std::function<void(const JetMatrixX1 &, const JetMatrixX1 &, JetMatrixX1 &)>;

  ErrorStateEkf() = default;

  // inject 和 box_minus 要传泛型可调用体：它们会同时被 double 和 Jet 实例化，
  // 前者用于推进名义状态，后者用于求 F。
  template <class Inject, class BoxMinus>
  ErrorStateEkf(
    const PredictFunc & f, const UpdateQFunc & update_q, const Inject & inject,
    const BoxMinus & box_minus, const MatrixXX & p0)
  : f_(f), update_Q_(update_q), P_delta_(p0)
  {
    setInject(inject);
    setBoxMinus(box_minus);
  }

  void setState(const MatrixX1 & x0) noexcept
  {
    x_nominal_ = x0;
    delta_x_.setZero();
  }

  void setUpdateQ(const UpdateQFunc & update_q) { update_Q_ = update_q; }
  void setPredictFunc(const PredictFunc & f) { f_ = f; }
  void setIterationNum(int n) { iteration_num_ = std::max(1, n); }

  // 选 updateMulti 里用哪种迭代式，默认教科书形式：
  //
  //   教科书（Bell & Cathey）：δ ← K·(r + H·δ)   每轮把 δ 重新锚回先验
  //   累加式：                 δ ← δ + K·r       每轮只累加高斯牛顿步
  //
  // 两者差 (I − KH)·δ。迭代次数一多，累加式的偏差会复利放大：3m_run_fast 上
  // 迭代次数从 1 升到 5，车心帧间跳变的 p99 从 0.126 m 单调恶化到 0.326 m。
  // 详见 docs/iterated_ekf.md。
  void setTextbook(bool enabled) { textbook_iteration_ = enabled; }

  template <class Inject>
  void setInject(const Inject & inject)
  {
    inject_state_ = inject;
    inject_state_jet_ = inject;
  }

  template <class BoxMinus>
  void setBoxMinus(const BoxMinus & box_minus)
  {
    box_minus_state_ = box_minus;
    box_minus_state_jet_ = box_minus;
  }

  const MatrixX1 & state() const noexcept { return x_nominal_; }
  const MatrixXX & covariance() const noexcept { return P_delta_; }

  // 预测步：名义状态按 x̌⁺ = f(x̌) 推进，再用 Jet 求 F 并传播 P。
  //
  // F 是 ∂δ⁺/∂δ，而 δ 与 δ⁺ 住在不同点的切空间里，不能直接对状态求导，要绕
  // 这条链：
  //
  //   δ ──⊞──▶ x̌ ⊞ δ ──f──▶ f(x̌ ⊞ δ) ──⊟──▶ δ⁺
  //
  // 把 δ 播种成单位阵推过去，出口每一行的导数就是 F。链式法则由 Jet 完成，
  // ∂⊞/∂δ、∂f/∂x、∂⊟/∂x 三段自动相乘，右雅可比就是这样被吸收掉的，不用显式算。
  MatrixX1 predict() noexcept
  {
    const MatrixX1 x_prev = x_nominal_;
    MatrixX1 x_pred;
    f_(x_prev.data(), x_pred.data());
    x_nominal_ = x_pred;

    JetMatrixX1 x_nominal_jet;
    JetMatrixX1 x_prev_jet;
    for (int i = 0; i < N_X; ++i) {
      x_nominal_jet[i] = Jet(x_nominal_[i]);
      x_prev_jet[i] = Jet(x_prev[i]);
    }

    JetMatrixX1 delta_jet;
    for (int i = 0; i < N_X; ++i) {
      delta_jet[i] = Jet(0.0, i);  // 单位阵播种
    }

    JetMatrixX1 x_pert_jet = x_prev_jet;
    inject_state_jet_(delta_jet, x_pert_jet);

    JetMatrixX1 x_pert_pred_jet;
    f_(x_pert_jet.data(), x_pert_pred_jet.data());

    JetMatrixX1 delta_pred_jet;
    box_minus_state_jet_(x_nominal_jet, x_pert_pred_jet, delta_pred_jet);

    for (int i = 0; i < N_X; ++i) {
      F_.row(i) = delta_pred_jet[i].v.transpose();
    }

    // 正常流程里 delta_x_ 在每次更新末尾清零，这一行乘的是零向量；留着是为了
    // 支持“连预测多次再更新一次”的用法。
    delta_x_ = F_ * delta_x_;

    Q_ = update_Q_();
    P_delta_ = F_ * P_delta_ * F_.transpose() + Q_;
    P_delta_ = 0.5 * (P_delta_ + P_delta_.transpose());  // 强制对称，抗数值漂移

    return x_nominal_;
  }

  // 一个观测的类型擦除接口：给出观测维数、预测值和残差。它只认 VectorXd，
  // Jet 流不过去，所以 updateMulti 用中心差分求 H。
  struct ObsBase
  {
    virtual ~ObsBase() = default;
    virtual int dim() const = 0;
    virtual void predict(const Eigen::VectorXd & x, Eigen::VectorXd & z_pred) const = 0;
    virtual void residualAndR(
      const Eigen::VectorXd & z_pred, Eigen::VectorXd & residual,
      Eigen::MatrixXd & r) const = 0;
  };

  template <int N_Z, class MeasureFunc, class UpdateRFunc, class ResidualFunc>
  struct ObsImpl : public ObsBase
  {
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    MatrixZ1 z;
    MeasureFunc h;
    UpdateRFunc update_R;
    ResidualFunc residual_func;

    ObsImpl(const MatrixZ1 & z_in, MeasureFunc h_in, UpdateRFunc r_in, ResidualFunc res_in)
    : z(z_in), h(std::move(h_in)), update_R(std::move(r_in)), residual_func(std::move(res_in))
    {
    }

    int dim() const override { return N_Z; }

    void predict(const Eigen::VectorXd & x, Eigen::VectorXd & z_pred) const override
    {
      assert(x.size() == N_X);
      std::array<double, N_X> x_data;
      for (int i = 0; i < N_X; ++i) {
        x_data[i] = x[i];
      }
      std::array<double, N_Z> z_data;
      h(x_data.data(), z_data.data());

      z_pred.resize(N_Z);
      for (int i = 0; i < N_Z; ++i) {
        z_pred[i] = z_data[i];
      }
    }

    void residualAndR(
      const Eigen::VectorXd & z_pred, Eigen::VectorXd & residual,
      Eigen::MatrixXd & r) const override
    {
      assert(z_pred.size() == N_Z);
      MatrixZ1 z_pred_fixed;
      for (int i = 0; i < N_Z; ++i) {
        z_pred_fixed[i] = z_pred[i];
      }
      residual = residual_func(z_pred_fixed, z);
      r = update_R(z);
    }
  };

  template <int N_Z, class MeasureFunc, class UpdateRFunc, class ResidualFunc>
  static std::shared_ptr<ObsBase> makeObs(
    const Eigen::Matrix<double, N_Z, 1> & z, MeasureFunc && h, UpdateRFunc && r,
    ResidualFunc && res)
  {
    using ObsT = ObsImpl<
      N_Z, std::decay_t<MeasureFunc>, std::decay_t<UpdateRFunc>, std::decay_t<ResidualFunc>>;
    return std::make_shared<ObsT>(
      z, std::forward<MeasureFunc>(h), std::forward<UpdateRFunc>(r),
      std::forward<ResidualFunc>(res));
  }

  // 多观测迭代更新：把本帧所有观测垂直拼成一个大的 [H; residual; R]，迭代
  // iteration_num 轮高斯牛顿，最后用 Joseph 形式更新 P 并把 δ 注入名义状态。
  //
  // R 是块对角，所以信息严格相加：P₊⁻¹ = P₋⁻¹ + Σ Hₖᵀ Rₖ⁻¹ Hₖ。一块完整装甲板
  // 拆成两条灯条共 8 行，每条独立灯条再加 4 行。
  MatrixX1 updateMulti(const std::vector<std::shared_ptr<ObsBase>> & obs_list) noexcept
  {
    int total_dim = 0;
    for (const auto & obs : obs_list) {
      total_dim += obs->dim();
    }
    if (total_dim == 0) {
      return x_nominal_;
    }

    Eigen::MatrixXd h_matrix(total_dim, N_X);
    Eigen::VectorXd residual(total_dim);
    Eigen::MatrixXd r_matrix = Eigen::MatrixXd::Zero(total_dim, total_dim);

    MatrixX1 delta_iter = delta_x_;
    MatrixXX p_iter = P_delta_;  // 迭代中不更新
    Eigen::MatrixXd k_matrix(N_X, total_dim);

    for (int iter = 0; iter < iteration_num_; ++iter) {
      // 每轮在当前迭代点 x̌ ⊞ δ_iter 重新线性化。
      int offset = 0;
      for (const auto & obs : obs_list) {
        const int d = obs->dim();

        Eigen::VectorXd rk;
        Eigen::MatrixXd rk_cov;
        Eigen::MatrixXd hk;
        linearize(*obs, delta_iter, rk, rk_cov, hk);

        h_matrix.block(offset, 0, d, N_X) = hk;
        residual.segment(offset, d) = rk;
        r_matrix.block(offset, offset, d, d) = rk_cov;
        offset += d;
      }

      const Eigen::MatrixXd s_matrix =
        h_matrix * p_iter * h_matrix.transpose() + r_matrix;
      // 创新量只在第 0 轮记：这一轮在先验点线性化，r 和 S 才是卡方检验要的
      // 那一对。后面几轮的残差已被迭代压缩，拿它算 NIS 会系统性偏小。
      if (iter == 0) {
        last_residual_ = residual;
        last_innovation_covariance_ = s_matrix;
      }
      const auto ldlt = s_matrix.ldlt();
      const Eigen::MatrixXd pht = p_iter * h_matrix.transpose();
      k_matrix = ldlt.solve(pht.transpose()).transpose();  // K = P Hᵀ S⁻¹

      if (textbook_iteration_) {
        // 教科书形式：每轮把 δ 重新锚回先验，迭代才真正是在同一个 MAP 目标上
        // 做高斯牛顿，而不是把先验项反复计入。
        delta_iter = k_matrix * (residual + h_matrix * delta_iter);
      } else {
        // 累加式，保留以便与旧结果对拍。
        delta_iter.noalias() += k_matrix * residual;
      }
    }

    inject_state_(delta_iter, x_nominal_);
    delta_x_.setZero();

    // Joseph 形式，对任意 K 都保持半正定。迭代 EKF 用的是最后一轮的 K，本来
    // 就不是最优增益，所以这里不是可选优化而是必需。
    const MatrixXX identity = MatrixXX::Identity();
    const Eigen::MatrixXd ikh = identity - k_matrix * h_matrix;
    P_delta_ = ikh * p_iter * ikh.transpose() +
               k_matrix * r_matrix * k_matrix.transpose();
    P_delta_ = 0.5 * (P_delta_ + P_delta_.transpose());

    return x_nominal_;
  }

  // 单个观测在当前先验点上的创新量 r 和协方差 S = H·P·Hᵀ + R，不改任何状态。
  // 给关联门限算马氏距离用，调用前应已 predict 到观测时刻。
  void innovation(
    const ObsBase & obs, Eigen::VectorXd & residual, Eigen::MatrixXd & covariance) const
  {
    Eigen::MatrixXd r_cov;
    Eigen::MatrixXd h;
    linearize(obs, delta_x_, residual, r_cov, h);
    covariance = h * P_delta_ * h.transpose() + r_cov;
  }

  // 最近一次更新在先验点上的创新量和它的协方差，供 NIS 记账与遥测读取。
  const Eigen::VectorXd & lastResidual() const noexcept { return last_residual_; }
  const Eigen::MatrixXd & lastInnovCov() const noexcept
  {
    return last_innovation_covariance_;
  }

private:
  // 在 x̌ ⊞ delta 处求一个观测的残差、R 和 H。H 用中心差分，三处关键：
  //   扰动加在 δ 上再 ⊞ 进去，求出的才是 ∂z/∂δ，与 P 同一坐标系；
  //   差的是 residual 而不是 z_pred，观测若带角度这类缠绕分量，归一化才进得到
  //   导数里，否则预测值分居 ±π 两侧时会差出一个 2π 的假梯度；
  //   末尾取负，因为 r = z - ẑ，所以 -∂r/∂δ = ∂ẑ/∂δ = H。
  void linearize(
    const ObsBase & obs, const MatrixX1 & delta, Eigen::VectorXd & residual,
    Eigen::MatrixXd & r_cov, Eigen::MatrixXd & h) const
  {
    MatrixX1 x_eval = x_nominal_;
    inject_state_(delta, x_eval);
    Eigen::VectorXd z_pred;
    obs.predict(x_eval, z_pred);
    obs.residualAndR(z_pred, residual, r_cov);

    constexpr double kEps = 1e-6;
    h.resize(obs.dim(), N_X);
    for (int i = 0; i < N_X; ++i) {
      MatrixX1 delta_plus = delta;
      MatrixX1 delta_minus = delta;
      delta_plus[i] += kEps;
      delta_minus[i] -= kEps;

      MatrixX1 x_plus = x_nominal_;
      MatrixX1 x_minus = x_nominal_;
      inject_state_(delta_plus, x_plus);
      inject_state_(delta_minus, x_minus);

      Eigen::VectorXd z_plus;
      Eigen::VectorXd z_minus;
      obs.predict(x_plus, z_plus);
      obs.predict(x_minus, z_minus);

      Eigen::VectorXd residual_plus;
      Eigen::VectorXd residual_minus;
      Eigen::MatrixXd ignored;
      obs.residualAndR(z_plus, residual_plus, ignored);
      obs.residualAndR(z_minus, residual_minus, ignored);

      h.col(i) = -(residual_plus - residual_minus) / (2.0 * kEps);
    }
  }

  PredictFunc f_{};
  UpdateQFunc update_Q_{};
  InjectFunc inject_state_{};
  BoxMinusFunc box_minus_state_{};
  InjectJetFunc inject_state_jet_{};
  BoxMinusJetFunc box_minus_state_jet_{};

  MatrixXX F_{MatrixXX::Zero()};
  MatrixXX Q_{MatrixXX::Zero()};

  MatrixX1 x_nominal_{MatrixX1::Zero()};
  MatrixX1 delta_x_{MatrixX1::Zero()};
  MatrixXX P_delta_{MatrixXX::Identity()};

  Eigen::VectorXd last_residual_{};
  Eigen::MatrixXd last_innovation_covariance_{};

  int iteration_num_{1};
  bool textbook_iteration_{true};
};

}  // namespace L3Estimation
