// 整车模型的几何、流形运算与误差状态转移矩阵检查。
//
// 三件事，重要性递增：
//   1. armorPose 生成的几何是否自洽（半径、间隔、板法向朝内）
//   2. ⊞ 与 ⊟ 是否严格互逆 —— 这一条挂了，后面全部无意义
//   3. F = ∂(f(x̌ ⊞ δ) ⊟ f(x̌))/∂δ 的 Jet 结果是否等于中心差分
//
// 第 2 条尤其要紧：⊟ 的转置位置若与 ⊞ 的右乘不配套，滤波器**照样能跑**，
// 只是 Jacobian 全错、协方差没有意义、遇到大机动才发散。没有这条断言，这类
// 错误要到实车上才暴露。

#include "l3_estimation/armor/vehicle_model.hpp"

#include <ceres/jet.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <numbers>
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

using State = Eigen::Matrix<double, VM::kStateSize, 1>;

// 一个"车在 (2.4, -1.1, 0.15)，正在平移并自转，姿态有小幅 roll/pitch"的状态。
State makeState()
{
  State x = State::Zero();
  x[VM::idx::CX] = 2.4;
  x[VM::idx::CY] = -1.1;
  x[VM::idx::CZ] = 0.15;
  x[VM::idx::VCX] = 0.8;
  x[VM::idx::VCY] = -0.35;
  x[VM::idx::VCZ] = 0.05;
  x[VM::idx::ROT_Z] = 0.62;
  x[VM::idx::ROT_Y] = -0.07;
  x[VM::idx::ROT_X] = 0.04;
  x[VM::idx::VYAW] = 3.1;
  x[VM::idx::LOG_R1] = std::log(0.26);
  x[VM::idx::LOG_R2] = std::log(0.29);
  x[VM::idx::HEIGHT] = 0.045;
  return x;
}

}  // namespace

int main()
{
  constexpr auto kName = L3Estimation::ArmorName::Infantry3;
  constexpr int kArmorNum = 4;
  const State state = makeState();

  // --- 1. 几何自洽 ---------------------------------------------------
  {
    const Eigen::Vector3d center(
      state[VM::idx::CX], state[VM::idx::CY], state[VM::idx::CZ]);

    for (int id = 0; id < kArmorNum; ++id) {
      const auto pose = VM::armorPose<double>(state.data(), id, kArmorNum, kName);

      // 板心到车心的水平距离必须等于该组半径。
      const double expected_radius =
        (id & 1) ? std::exp(state[VM::idx::LOG_R2]) : std::exp(state[VM::idx::LOG_R1]);
      const Eigen::Vector3d offset = pose.translation() - center;
      // 高度差沿车体 z 轴，先转回车体系再量水平距离。
      const Eigen::Matrix3d vehicle_rotation =
        VM::vehicleRotation<double>(state.data(), kName);
      const Eigen::Vector3d offset_in_vehicle = vehicle_rotation.transpose() * offset;
      expectNear(
        std::hypot(offset_in_vehicle.x(), offset_in_vehicle.y()), expected_radius, 1e-12,
        "板心到车心的水平距离不等于半径");

      // 奇数板抬高 HEIGHT。
      expectNear(
        offset_in_vehicle.z(), (id & 1) ? state[VM::idx::HEIGHT] : 0.0, 1e-12,
        "板的高度偏移不对");

      // 板的 x 轴指向车心：与"板心 → 车心"同向。
      const Eigen::Vector3d axis_x = pose.linear().col(0);
      const Eigen::Vector3d to_center = (center - pose.translation()).normalized();
      expect(axis_x.dot(to_center) > 0.9, "板的 x 轴没有指向车心");
    }

    // 相邻板的方位角间隔必须是 2π/N。
    for (int id = 0; id < kArmorNum; ++id) {
      const auto a = VM::armorPose<double>(state.data(), id, kArmorNum, kName);
      const auto b = VM::armorPose<double>(state.data(), (id + 1) % kArmorNum, kArmorNum, kName);
      const Eigen::Matrix3d relative = a.linear().transpose() * b.linear();
      const double angle = Eigen::AngleAxisd(relative).angle();
      expectNear(
        angle, 2.0 * std::numbers::pi / kArmorNum, 1e-12, "相邻板的朝向差不是 2π/N");
    }
  }

  // --- 2. ⊞ 与 ⊟ 严格互逆 --------------------------------------------
  //
  // (x̌ ⊞ δ) ⊟ x̌ == δ。随机取多组 δ，包括大到足以让 BCH 修正显形的量级——
  // 若 ⊟ 写成左乘形式，小 δ 下误差是二阶的、看不出来，必须用大 δ 才能抓住。
  {
    std::mt19937 rng(20260830);
    std::uniform_real_distribution<double> small(-0.05, 0.05);
    std::uniform_real_distribution<double> large(-0.8, 0.8);

    double max_error = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
      auto & dist = (trial % 2 == 0) ? small : large;
      State delta;
      for (int i = 0; i < VM::kStateSize; ++i) {
        delta[i] = dist(rng);
      }

      State perturbed = state;
      VM::injectState(delta, perturbed);

      State recovered;
      VM::boxMinusState(state, perturbed, recovered);

      max_error = std::max(max_error, (recovered - delta).cwiseAbs().maxCoeff());
    }
    expect(max_error < 1e-12, "(x̌ ⊞ δ) ⊟ x̌ != δ，⊞/⊟ 不互逆");
    if (max_error >= 1e-12) {
      std::cerr << "  max_error = " << max_error << '\n';
    }
  }

  // --- 3. 反向：x̌ ⊞ (x ⊟ x̌) == x ------------------------------------
  {
    State other = makeState();
    other[VM::idx::CX] += 0.7;
    other[VM::idx::ROT_Z] += 0.4;
    other[VM::idx::ROT_X] -= 0.15;
    other[VM::idx::VYAW] -= 1.2;

    State delta;
    VM::boxMinusState(state, other, delta);
    State reconstructed = state;
    VM::injectState(delta, reconstructed);

    // 旋转向量本身可能落在不同分支，所以比旋转矩阵而不是比裸分量。
    const Eigen::Matrix3d expected = VM::stateRotation(other);
    const Eigen::Matrix3d actual = VM::stateRotation(reconstructed);
    expect((expected - actual).norm() < 1e-12, "x̌ ⊞ (x ⊟ x̌) 的姿态与 x 不符");

    for (int i = 0; i < VM::kStateSize; ++i) {
      if (i == VM::idx::ROT_X || i == VM::idx::ROT_Y || i == VM::idx::ROT_Z) {
        continue;
      }
      expectNear(reconstructed[i], other[i], 1e-12, "x̌ ⊞ (x ⊟ x̌) 的欧氏分量与 x 不符");
    }
  }

  // --- 4. 误差状态转移矩阵 F：Jet 对拍中心差分 ------------------------
  //
  // 这是整个移植真正要守住的东西。F 不是 ∂f/∂x，而是 ∂δ⁺/∂δ，两个 δ 住在
  // 不同点的切空间里，所以必须绕 ⊞ → f → ⊟ 一圈。
  {
    using Jet = ceres::Jet<double, VM::kStateSize>;
    using JetState = Eigen::Matrix<Jet, VM::kStateSize, 1>;

    const VM::Motion motion{.dt = 0.02, .name = kName};

    // 名义状态推进一步，作为 ⊟ 的基准点。
    State nominal_next;
    motion(state.data(), nominal_next.data());

    // δ 播种单位阵，推过 ⊞ → f → ⊟。
    JetState delta_jet;
    JetState state_jet;
    JetState nominal_next_jet;
    for (int i = 0; i < VM::kStateSize; ++i) {
      delta_jet[i] = Jet(0.0, i);
      state_jet[i] = Jet(state[i]);
      nominal_next_jet[i] = Jet(nominal_next[i]);
    }

    JetState perturbed_jet = state_jet;
    VM::injectState(delta_jet, perturbed_jet);

    JetState perturbed_next_jet;
    motion(perturbed_jet.data(), perturbed_next_jet.data());

    JetState delta_next_jet;
    VM::boxMinusState(nominal_next_jet, perturbed_next_jet, delta_next_jet);

    Eigen::Matrix<double, VM::kStateSize, VM::kStateSize> jacobian;
    for (int i = 0; i < VM::kStateSize; ++i) {
      jacobian.row(i) = delta_next_jet[i].v.transpose();
    }

    // 同一个量用中心差分再算一遍。
    constexpr double kStep = 1e-6;
    double max_error = 0.0;
    for (int col = 0; col < VM::kStateSize; ++col) {
      State delta_plus = State::Zero();
      State delta_minus = State::Zero();
      delta_plus[col] = kStep;
      delta_minus[col] = -kStep;

      State plus = state;
      State minus = state;
      VM::injectState(delta_plus, plus);
      VM::injectState(delta_minus, minus);

      State plus_next;
      State minus_next;
      motion(plus.data(), plus_next.data());
      motion(minus.data(), minus_next.data());

      State delta_plus_next;
      State delta_minus_next;
      VM::boxMinusState(nominal_next, plus_next, delta_plus_next);
      VM::boxMinusState(nominal_next, minus_next, delta_minus_next);

      for (int row = 0; row < VM::kStateSize; ++row) {
        const double numeric =
          (delta_plus_next[row] - delta_minus_next[row]) / (2.0 * kStep);
        max_error = std::max(max_error, std::abs(jacobian(row, col) - numeric));
      }
    }
    expect(max_error < 1e-6, "F 的 Jet 结果与中心差分不符");
    if (max_error >= 1e-6) {
      std::cerr << "  max_error = " << max_error << '\n';
    }

    // 恒速度模型的两个已知结构：位置对速度的偏导是 dt，速度对自己是 1。
    expectNear(jacobian(VM::idx::CX, VM::idx::VCX), 0.02, 1e-9, "∂cx/∂vcx 应为 dt");
    expectNear(jacobian(VM::idx::VCX, VM::idx::VCX), 1.0, 1e-9, "∂vcx/∂vcx 应为 1");
    expectNear(jacobian(VM::idx::ROT_Z, VM::idx::VYAW), 0.02, 1e-6, "∂rot_z/∂vyaw 应为 dt");
  }

  // --- 5. 前哨站退化：只估 yaw，半径钉死 ------------------------------
  {
    State outpost = State::Zero();
    outpost[VM::idx::CZ] = 1.2;
    outpost[VM::idx::ROT_Z] = 0.3;
    outpost[VM::idx::ROT_X] = 0.9;  // 应当被忽略
    outpost[VM::idx::LOG_R1] = std::log(0.9);  // 应当被 clamp 钉回规则半径

    const Eigen::Matrix3d rotation =
      VM::vehicleRotation<double>(outpost.data(), L3Estimation::ArmorName::Outpost);
    const Eigen::Vector3d logged = L3Estimation::so3Log<double>(rotation);
    expect(
      std::abs(logged.x()) < 1e-12 && std::abs(logged.y()) < 1e-12,
      "前哨站姿态应当只有 yaw 分量");

    const VM::Motion motion{.dt = 0.01, .name = L3Estimation::ArmorName::Outpost};
    State next;
    motion(outpost.data(), next.data());
    expectNear(
      std::exp(next[VM::idx::LOG_R1]), VM::kOutpostRadius, 1e-12,
      "前哨站半径应当被钉死为规则值");
  }

  // --- 6. 过程噪声：体系构造、常加速度结构、log 半径换算 --------------
  {
    VM::NoiseConfig config;
    config.body_acceleration = Eigen::Vector3d{30.0, 10.0, 1.0};  // 三轴刻意不同
    config.yaw_acceleration = 25.0;
    config.radius = 1e-6;
    config.roll_pitch = 0.2;

    constexpr double kDt = 0.02;

    // 车头朝世界 x（yaw = 0）时，体系与世界系重合。
    State aligned = State::Zero();
    aligned[VM::idx::LOG_R1] = std::log(0.26);
    aligned[VM::idx::LOG_R2] = std::log(0.30);
    const auto q_aligned = VM::processNoise(aligned, kDt, kName, config);

    const double dt2 = kDt * kDt;
    const double dt3 = dt2 * kDt;
    const double dt4 = dt2 * dt2;

    // 常加速度模型的四个系数。
    expectNear(
      q_aligned(VM::idx::CX, VM::idx::CX), 0.25 * dt4 * 30.0, 1e-15, "Q 位置-位置块系数错");
    expectNear(
      q_aligned(VM::idx::CX, VM::idx::VCX), 0.5 * dt3 * 30.0, 1e-15, "Q 位置-速度块系数错");
    expectNear(
      q_aligned(VM::idx::VCX, VM::idx::VCX), dt2 * 30.0, 1e-15, "Q 速度-速度块系数错");
    expectNear(
      q_aligned(VM::idx::VCY, VM::idx::VCY), dt2 * 10.0, 1e-15, "Q 的 y 轴强度没有独立生效");
    expectNear(
      q_aligned(VM::idx::ROT_Z, VM::idx::VYAW), 0.5 * dt3 * 25.0, 1e-15, "Q 的 yaw 耦合块错");

    // 车头转 90 度后，体系 x 的强度应当出现在世界 y 上 —— 这是"体系 Q 再旋转"
    // 的全部意义。若直接在世界系给对角阵，这个测试必然失败。
    State turned = aligned;
    turned[VM::idx::ROT_Z] = std::numbers::pi / 2.0;
    const auto q_turned = VM::processNoise(turned, kDt, kName, config);
    expectNear(
      q_turned(VM::idx::VCY, VM::idx::VCY), dt2 * 30.0, 1e-12,
      "车头转 90 度后体系 x 的噪声没有旋到世界 y");
    expectNear(
      q_turned(VM::idx::VCX, VM::idx::VCX), dt2 * 10.0, 1e-12,
      "车头转 90 度后体系 y 的噪声没有旋到世界 x");

    // log 半径的噪声换算：q_ℓℓ = q_r / r²
    expectNear(
      q_aligned(VM::idx::LOG_R1, VM::idx::LOG_R1), 1e-6 / (0.26 * 0.26), 1e-15,
      "LOG_R1 的噪声没有按 1/r² 换算");
    expectNear(
      q_aligned(VM::idx::LOG_R2, VM::idx::LOG_R2), 1e-6 / (0.30 * 0.30), 1e-15,
      "LOG_R2 的噪声没有按 1/r² 换算");
    expect(
      q_aligned(VM::idx::LOG_R2, VM::idx::LOG_R2) <
        q_aligned(VM::idx::LOG_R1, VM::idx::LOG_R1),
      "大半径目标的对数噪声应当更小");

    // roll/pitch 是随机游走，yaw 那一维不重复计入。
    expectNear(
      q_aligned(VM::idx::ROT_X, VM::idx::ROT_X), kDt * 0.2, 1e-15, "roll 噪声错");
    expect(
      q_aligned(VM::idx::ROT_Z, VM::idx::ROT_Z) > 0.0 &&
        std::abs(q_aligned(VM::idx::ROT_Z, VM::idx::ROT_Z) - 0.25 * dt4 * 25.0) < 1e-15,
      "yaw 方向不应叠加 roll_pitch 的随机游走");

    // 对称半正定。
    expect(
      (q_turned - q_turned.transpose()).cwiseAbs().maxCoeff() < 1e-15, "Q 不对称");
    const Eigen::MatrixXd q_dynamic = q_turned;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver{q_dynamic};
    expect(solver.eigenvalues().minCoeff() > -1e-15, "Q 不是半正定");

    // 前哨站走另一组参数。
    State outpost = State::Zero();
    outpost[VM::idx::LOG_R1] = std::log(VM::kOutpostRadius);
    const auto q_outpost = VM::processNoise(
      outpost, kDt, L3Estimation::ArmorName::Outpost, config);
    expect(
      q_outpost(VM::idx::VYAW, VM::idx::VYAW) < q_aligned(VM::idx::VYAW, VM::idx::VYAW),
      "前哨站的角加速度噪声应当远小于常规车");
  }

  if (failure_count != 0) {
    std::cerr << "vehicle model smoke test failed with " << failure_count << " error(s)\n";
    return 1;
  }
  std::cout << "vehicle model smoke test passed\n";
  return 0;
}
