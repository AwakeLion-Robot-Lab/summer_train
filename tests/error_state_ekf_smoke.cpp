// 误差状态 EKF 的结构正确性检查。
//
// 两件事：
//   1. 退化对拍 —— 把 ⊞/⊟ 换成普通加减、系统换成线性，滤波器必须逐位等于
//      教科书 KF。这条守住的是"卡尔曼那部分没写错"，与流形无关。
//   2. 流形收敛 —— 在真正的整车模型上，用装甲板三维位置观测，从偏离真值的
//      初值出发，看状态是否收敛回去。这条守住的是 ⊞/⊟ 与数值 H 接得上。
//
// 第 1 条之所以必要：ESEKF 里任何一处符号错误都不会让程序崩，只会让它慢慢
// 收敛到错的地方。先用一个有解析解的场景把卡尔曼骨架钉死，出问题时才能确定
// 是流形那半边的错。

#include "l3_estimation/armor/vehicle_model.hpp"
#include "l3_estimation/error_state_ekf.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <random>
#include <string_view>

namespace VM = L3Estimation::VehicleModel;

namespace {

int failure_count = 0;

void expect(bool condition, std::string_view message)
{
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failure_count;
  }
}

void expectNear(double actual, double expected, double tolerance, std::string_view message)
{
  if (!(std::abs(actual - expected) <= tolerance)) {
    std::cerr << "FAIL: " << message << "  actual=" << actual << " expected=" << expected
              << " diff=" << std::abs(actual - expected) << '\n';
    ++failure_count;
  }
}

// --- 场景一：一维匀速运动，状态 [位置, 速度] ---------------------------

constexpr double kDt = 0.1;

struct LinearMotion
{
  template <typename T>
  void operator()(const T * x0, T * x1) const
  {
    x1[0] = x0[0] + x0[1] * T(kDt);
    x1[1] = x0[1];
  }
};

}  // namespace

int main()
{
  // ================= 1. 退化成教科书 KF =================
  {
    using Filter = L3Estimation::ErrorStateEkf<2, LinearMotion>;
    using Vector2 = Eigen::Matrix<double, 2, 1>;
    using Matrix2 = Eigen::Matrix<double, 2, 2>;

    // ⊞ 和 ⊟ 退化成普通加减，此时误差状态与状态重合。
    const auto inject = [](const auto & delta, auto & nominal) {
      for (int i = 0; i < 2; ++i) {
        nominal[i] += delta[i];
      }
    };
    const auto box_minus = [](const auto & nominal, const auto & value, auto & delta) {
      delta = value - nominal;
    };

    Matrix2 q;
    q << 1e-4, 0.0, 0.0, 1e-3;
    Matrix2 p0;
    p0 << 1.0, 0.0, 0.0, 4.0;

    Filter filter(LinearMotion{}, [&]() { return q; }, inject, box_minus, p0);
    Vector2 x0;
    x0 << 0.0, 0.0;
    filter.setState(x0);
    filter.setIterationNum(1);  // 线性系统迭代无意义，取 1 才能与 KF 逐位对齐

    // 手写一份标准 KF 作为参照。
    Eigen::Matrix<double, 2, 1> kf_x = x0;
    Matrix2 kf_p = p0;
    Matrix2 kf_f;
    kf_f << 1.0, kDt, 0.0, 1.0;
    Eigen::Matrix<double, 1, 2> kf_h;
    kf_h << 1.0, 0.0;
    Eigen::Matrix<double, 1, 1> kf_r;
    kf_r << 0.25;

    // 观测：直接看位置。
    const auto measure = [](const double * x, double * z) { z[0] = x[0]; };
    const auto update_r = [&](const Eigen::Matrix<double, 1, 1> &) { return kf_r; };
    const auto residual = [](
                            const Eigen::Matrix<double, 1, 1> & z_pred,
                            const Eigen::Matrix<double, 1, 1> & z) { return z - z_pred; };

    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 0.5);
    double truth_position = 0.0;
    constexpr double kTruthVelocity = 1.7;

    double max_state_error = 0.0;
    double max_covariance_error = 0.0;
    for (int step = 0; step < 40; ++step) {
      truth_position += kTruthVelocity * kDt;
      Eigen::Matrix<double, 1, 1> z;
      z << truth_position + noise(rng);

      // --- ESEKF ---
      filter.predict();
      std::vector<std::shared_ptr<Filter::ObsBase>> obs;
      obs.push_back(Filter::makeObs<1>(z, measure, update_r, residual));
      filter.updateMulti(obs);

      // --- 教科书 KF ---
      kf_x = kf_f * kf_x;
      kf_p = kf_f * kf_p * kf_f.transpose() + q;
      const Eigen::Matrix<double, 1, 1> innovation = z - kf_h * kf_x;
      const Eigen::Matrix<double, 1, 1> s = kf_h * kf_p * kf_h.transpose() + kf_r;
      const Eigen::Matrix<double, 2, 1> k = kf_p * kf_h.transpose() * s.inverse();
      kf_x += k * innovation;
      const Matrix2 ikh = Matrix2::Identity() - k * kf_h;
      kf_p = ikh * kf_p * ikh.transpose() + k * kf_r * k.transpose();

      max_state_error =
        std::max(max_state_error, (filter.state() - kf_x).cwiseAbs().maxCoeff());
      max_covariance_error =
        std::max(max_covariance_error, (filter.covariance() - kf_p).cwiseAbs().maxCoeff());
    }

    // 差异只应来自 H 的中心差分（线性观测下截断误差为零，只剩舍入）。
    expect(max_state_error < 1e-9, "线性退化下状态与教科书 KF 不符");
    expect(max_covariance_error < 1e-9, "线性退化下协方差与教科书 KF 不符");
    if (max_state_error >= 1e-9 || max_covariance_error >= 1e-9) {
      std::cerr << "  state=" << max_state_error << " cov=" << max_covariance_error << '\n';
    }

    // 顺带确认它确实在跟踪：位置误差应当远小于观测噪声。
    expectNear(filter.state()[1], kTruthVelocity, 0.35, "线性场景下速度没有收敛");
  }

  // ================= 2. 整车模型上的收敛 =================
  //
  // 观测取四块装甲板在世界系的三维位置（共 12 维）。这不是最终要用的 UVL
  // 观测，但足以验证 ⊞/⊟ 与数值 H 在真实几何上接得上。
  {
    constexpr auto kName = L3Estimation::ArmorName::Infantry3;
    constexpr int kArmorNum = 4;
    constexpr int kZ = 3 * kArmorNum;
    using Filter = L3Estimation::ErrorStateEkf<VM::kStateSize, VM::Motion>;
    using State = Eigen::Matrix<double, VM::kStateSize, 1>;
    using MeasZ = Eigen::Matrix<double, kZ, 1>;

    State truth = State::Zero();
    truth[VM::idx::CX] = 3.0;
    truth[VM::idx::CY] = 0.6;
    truth[VM::idx::CZ] = 0.12;
    truth[VM::idx::ROT_Z] = 0.35;
    truth[VM::idx::VYAW] = 2.2;
    truth[VM::idx::LOG_R1] = std::log(0.26);
    truth[VM::idx::LOG_R2] = std::log(0.28);
    truth[VM::idx::HEIGHT] = 0.05;

    // 初值故意偏掉：车心偏 15cm，yaw 偏 0.25rad，角速度完全不知道。
    State initial = truth;
    initial[VM::idx::CX] += 0.15;
    initial[VM::idx::CY] -= 0.10;
    initial[VM::idx::ROT_Z] += 0.25;
    initial[VM::idx::VYAW] = 0.0;
    initial[VM::idx::LOG_R1] = std::log(0.24);
    initial[VM::idx::LOG_R2] = std::log(0.24);
    initial[VM::idx::HEIGHT] = 0.0;

    const auto inject = [](const auto & delta, auto & nominal) {
      VM::injectState(delta, nominal);
    };
    const auto box_minus = [](const auto & nominal, const auto & value, auto & delta) {
      VM::boxMinusState(nominal, value, delta);
    };

    Eigen::Matrix<double, VM::kStateSize, VM::kStateSize> p0;
    p0.setZero();
    p0.diagonal().setConstant(1.0);
    p0.diagonal()[VM::idx::VYAW] = 100.0;  // 单帧看不出转速，给足不确定性

    Eigen::Matrix<double, VM::kStateSize, VM::kStateSize> q;
    q.setZero();
    q.diagonal().setConstant(1e-6);
    q.diagonal()[VM::idx::VCX] = q.diagonal()[VM::idx::VCY] = 1e-3;
    q.diagonal()[VM::idx::VYAW] = 1e-2;

    Filter filter(
      VM::Motion{.dt = 0.01, .name = kName}, [&]() { return q; }, inject, box_minus, p0);
    filter.setState(initial);
    filter.setIterationNum(3);

    // 观测函数：从状态展开四块板的世界系三维位置。
    const auto measure = [](const double * x, double * z) {
      for (int id = 0; id < kArmorNum; ++id) {
        const auto pose = VM::armorPose<double>(x, id, kArmorNum, kName);
        z[id * 3 + 0] = pose.translation().x();
        z[id * 3 + 1] = pose.translation().y();
        z[id * 3 + 2] = pose.translation().z();
      }
    };
    Eigen::Matrix<double, kZ, kZ> r_cov = Eigen::Matrix<double, kZ, kZ>::Zero();
    r_cov.diagonal().setConstant(1e-4);  // 1cm 量级
    const auto update_r = [&](const MeasZ &) { return r_cov; };
    const auto residual = [](const MeasZ & z_pred, const MeasZ & z) -> MeasZ {
      return z - z_pred;
    };

    State truth_now = truth;
    const VM::Motion truth_motion{.dt = 0.01, .name = kName};

    for (int step = 0; step < 300; ++step) {
      State next;
      truth_motion(truth_now.data(), next.data());
      truth_now = next;

      MeasZ z;
      measure(truth_now.data(), z.data());

      filter.predict();
      std::vector<std::shared_ptr<Filter::ObsBase>> obs;
      obs.push_back(Filter::makeObs<kZ>(z, measure, update_r, residual));
      filter.updateMulti(obs);
    }

    const State estimate = filter.state();

    // 整车模型对 yaw 有 2π/N 的对称性，编号相位可能与真值差一格，所以断言写在
    // **装甲板集合**上，而不是裸 yaw 上。
    double max_armor_error = 0.0;
    for (int id = 0; id < kArmorNum; ++id) {
      const auto truth_pose =
        VM::armorPose<double>(truth_now.data(), id, kArmorNum, kName);
      double best = std::numeric_limits<double>::max();
      for (int other = 0; other < kArmorNum; ++other) {
        const auto estimate_pose =
          VM::armorPose<double>(estimate.data(), other, kArmorNum, kName);
        best = std::min(
          best, (truth_pose.translation() - estimate_pose.translation()).norm());
      }
      max_armor_error = std::max(max_armor_error, best);
    }
    expect(max_armor_error < 5e-3, "整车模型上装甲板位置没有收敛到真值");
    if (max_armor_error >= 5e-3) {
      std::cerr << "  max_armor_error = " << max_armor_error << " m\n";
    }

    // 车心与角速度是直接可比的，不受编号相位影响。
    expectNear(estimate[VM::idx::CX], truth_now[VM::idx::CX], 5e-3, "车心 x 未收敛");
    expectNear(estimate[VM::idx::CY], truth_now[VM::idx::CY], 5e-3, "车心 y 未收敛");
    expectNear(
      std::abs(estimate[VM::idx::VYAW]), std::abs(truth[VM::idx::VYAW]), 0.15,
      "角速度未收敛");

    // 协方差必须保持对称半正定 —— Joseph form 的意义就在这。
    const Eigen::MatrixXd covariance = filter.covariance();
    expect(
      (covariance - covariance.transpose()).cwiseAbs().maxCoeff() < 1e-12,
      "协方差不对称");
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    expect(solver.eigenvalues().minCoeff() > -1e-12, "协方差不是半正定");
  }

  if (failure_count != 0) {
    std::cerr << "error state ekf smoke test failed with " << failure_count << " error(s)\n";
    return 1;
  }
  std::cout << "error state ekf smoke test passed\n";
  return 0;
}
