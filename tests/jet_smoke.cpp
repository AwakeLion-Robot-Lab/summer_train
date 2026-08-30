// 前向自动微分与 SO(3) 映射的正确性检查。
//
// Jet 的每一条求导规则都拿中心差分对拍。这一步看着琐碎，但它是后面所有
// Jacobian 的地基：inject/box_minus 的符号错误、左右乘搞反、小角度分支写
// 错，最终都表现为某个 Jacobian 与数值微分对不上。先把地基钉死，后面出
// 问题时才能确定不是这一层的错。
//
// 数值微分的精度上限约为 eps^(2/3) ≈ 6e-6（中心差分的截断误差与舍入误差
// 折中），所以断言门限取 1e-6 量级而不是机器精度。

#include "l3_estimation/jet.hpp"
#include "l3_estimation/so3.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>

#include <cmath>
#include <functional>
#include <iostream>
#include <string_view>

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

constexpr int kDim = 4;
using Jet = L3Estimation::Jet<kDim>;
using Vector = Eigen::Matrix<double, kDim, 1>;

// 中心差分求梯度，作为自动微分的参照物。
Vector numericGradient(const std::function<double(const Vector &)> & f, const Vector & x)
{
  constexpr double kStep = 1e-6;
  Vector gradient = Vector::Zero();
  for (int i = 0; i < kDim; ++i) {
    Vector plus = x;
    Vector minus = x;
    plus[i] += kStep;
    minus[i] -= kStep;
    gradient[i] = (f(plus) - f(minus)) / (2.0 * kStep);
  }
  return gradient;
}

// 把 x 播种成一组 Jet：第 i 维带上第 i 个导数方向。
std::array<Jet, kDim> seedAll(const Vector & x)
{
  std::array<Jet, kDim> jets;
  for (int i = 0; i < kDim; ++i) {
    jets[i] = Jet::seed(x[i], i);
  }
  return jets;
}

void checkScalarFunction(
  const std::function<Jet(const std::array<Jet, kDim> &)> & jet_f,
  const std::function<double(const Vector &)> & double_f, const Vector & x,
  std::string_view name)
{
  const Jet value = jet_f(seedAll(x));
  const Vector numeric = numericGradient(double_f, x);

  expectNear(value.a, double_f(x), 1e-12, std::string(name) + ": 函数值与 double 版不一致");
  for (int i = 0; i < kDim; ++i) {
    expectNear(
      value.v[i], numeric[i], 1e-5,
      std::string(name) + ": 第 " + std::to_string(i) + " 维导数与中心差分不符");
  }
}

}  // namespace

int main()
{
  using L3Estimation::so3Exp;
  using L3Estimation::so3Log;
  using L3Estimation::so3Hat;

  // --- 1. 每个初等函数单独对拍 ---------------------------------------
  const Vector sample{0.7, -0.35, 1.4, 0.22};

  checkScalarFunction(
    [](const auto & j) { return j[0] * j[1] + j[2] / j[3]; },
    [](const Vector & x) { return x[0] * x[1] + x[2] / x[3]; }, sample, "乘除");

  checkScalarFunction(
    [](const auto & j) { return L3Estimation::sqrt(j[2] * j[2] + j[0] * j[0]); },
    [](const Vector & x) { return std::sqrt(x[2] * x[2] + x[0] * x[0]); }, sample, "sqrt");

  checkScalarFunction(
    [](const auto & j) { return L3Estimation::exp(j[0]) * L3Estimation::log(j[2]); },
    [](const Vector & x) { return std::exp(x[0]) * std::log(x[2]); }, sample, "exp/log");

  checkScalarFunction(
    [](const auto & j) { return L3Estimation::sin(j[1]) * L3Estimation::cos(j[3]); },
    [](const Vector & x) { return std::sin(x[1]) * std::cos(x[3]); }, sample, "sin/cos");

  checkScalarFunction(
    [](const auto & j) { return L3Estimation::atan2(j[0], j[2]); },
    [](const Vector & x) { return std::atan2(x[0], x[2]); }, sample, "atan2");

  checkScalarFunction(
    [](const auto & j) { return L3Estimation::acos(j[3]); },
    [](const Vector & x) { return std::acos(x[3]); }, sample, "acos");

  // 复合表达式：把整条链路会用到的算子串起来，抓单独测不出的组合错误。
  checkScalarFunction(
    [](const auto & j) {
      return L3Estimation::atan2(
               L3Estimation::exp(j[0]) * L3Estimation::sin(j[1]),
               L3Estimation::sqrt(j[2] * j[2] + Jet(1.0))) *
             L3Estimation::cos(j[3]);
    },
    [](const Vector & x) {
      return std::atan2(std::exp(x[0]) * std::sin(x[1]), std::sqrt(x[2] * x[2] + 1.0)) *
             std::cos(x[3]);
    },
    sample, "复合表达式");

  // --- 2. floor 的导数必须恒为零 -------------------------------------
  //
  // normalize_angle 依赖这一点：加减 2π 不改变角度的物理含义，导数也不该变。
  {
    const Jet x = Jet::seed(3.7, 0);
    const Jet y = L3Estimation::floor(x);
    expectNear(y.a, 3.0, 1e-15, "floor 函数值错误");
    expect(y.v.isZero(), "floor 的导数必须恒为零");
  }

  // --- 3. hat 算子等价于叉乘 -----------------------------------------
  {
    const Eigen::Vector3d w{0.3, -1.1, 0.45};
    const Eigen::Vector3d v{-0.7, 0.2, 1.3};
    expect((so3Hat<double>(w) * v - w.cross(v)).norm() < 1e-15, "so3Hat 与叉乘不等价");
  }

  // --- 4. so3Exp / so3Log 往返 ---------------------------------------
  {
    const Eigen::Vector3d angles[] = {
      {0.0, 0.0, 0.0},                 // 恒等，走泰勒分支
      {1e-9, -2e-9, 5e-10},            // 极小角，走泰勒分支
      {0.3, -0.2, 0.15},               // 常规
      {1.9, 0.4, -0.7},                // 大角，接近但不到 π
    };
    for (const auto & phi : angles) {
      const Eigen::Matrix3d rotation = so3Exp<double>(phi);
      expect(
        (rotation * rotation.transpose() - Eigen::Matrix3d::Identity()).norm() < 1e-12,
        "so3Exp 的结果不是正交阵");
      expectNear(rotation.determinant(), 1.0, 1e-12, "so3Exp 的行列式不是 +1");
      expect((so3Log<double>(rotation) - phi).norm() < 1e-9, "so3Log(so3Exp(φ)) != φ");
    }
  }

  // --- 5. 小角度分支的连续性 -----------------------------------------
  //
  // 阈值取 theta² < 1e-12，即 θ < 1e-6。在阈值两侧各取一点，两条公式算出的
  // 结果必须几乎一样，否则分支处会有跳变，Jacobian 在那里就是错的。
  {
    const Eigen::Vector3d axis = Eigen::Vector3d{1.0, -2.0, 0.5}.normalized();
    const Eigen::Matrix3d below = so3Exp<double>((0.9e-6 * axis).eval());
    const Eigen::Matrix3d above = so3Exp<double>((1.1e-6 * axis).eval());
    // 两点相差 2e-7 rad，旋转矩阵的差应当同量级。
    expect((above - below).norm() < 1e-6, "so3Exp 在小角度阈值两侧不连续");
  }

  // --- 6. so3Exp 的 Jacobian：Jet 对拍中心差分 ------------------------
  //
  // 这是本测试真正要守住的东西。ESEKF 的 F 就是靠 Jet 穿过 so3Exp / so3Log
  // 求出来的，这一步对了，后面才谈得上 inject/box_minus 对不对。
  {
    using Jet3 = L3Estimation::Jet<3>;
    const Eigen::Vector3d phi{0.42, -0.17, 0.31};

    Eigen::Matrix<Jet3, 3, 1> phi_jet;
    for (int i = 0; i < 3; ++i) {
      phi_jet[i] = Jet3::seed(phi[i], i);
    }
    const Eigen::Matrix<Jet3, 3, 3> rotation_jet = so3Exp<Jet3>(phi_jet);

    constexpr double kStep = 1e-6;
    double max_error = 0.0;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        for (int k = 0; k < 3; ++k) {
          Eigen::Vector3d plus = phi;
          Eigen::Vector3d minus = phi;
          plus[k] += kStep;
          minus[k] -= kStep;
          const double numeric =
            (so3Exp<double>(plus)(row, col) - so3Exp<double>(minus)(row, col)) / (2.0 * kStep);
          max_error = std::max(max_error, std::abs(rotation_jet(row, col).v[k] - numeric));
        }
      }
    }
    expect(max_error < 1e-6, "so3Exp 的 Jet Jacobian 与中心差分不符");
    if (max_error >= 1e-6) {
      std::cerr << "  max_error = " << max_error << '\n';
    }
  }

  // --- 7. so3Log 的 Jacobian：同样对拍 --------------------------------
  {
    using Jet3 = L3Estimation::Jet<3>;
    const Eigen::Vector3d phi{0.28, 0.36, -0.19};
    const Eigen::Matrix3d base = so3Exp<double>(phi);

    // 以 R·Exp(δ) 的形式扰动，δ 在零点求导——这正是 inject_state 的形状。
    Eigen::Matrix<Jet3, 3, 1> delta_jet;
    for (int i = 0; i < 3; ++i) {
      delta_jet[i] = Jet3::seed(0.0, i);
    }
    const Eigen::Matrix<Jet3, 3, 3> base_jet = base.cast<Jet3>();
    const Eigen::Matrix<Jet3, 3, 1> logged =
      so3Log<Jet3>((base_jet * so3Exp<Jet3>(delta_jet)).eval());

    constexpr double kStep = 1e-6;
    double max_error = 0.0;
    for (int row = 0; row < 3; ++row) {
      for (int k = 0; k < 3; ++k) {
        Eigen::Vector3d plus = Eigen::Vector3d::Zero();
        Eigen::Vector3d minus = Eigen::Vector3d::Zero();
        plus[k] += kStep;
        minus[k] -= kStep;
        const double numeric = (so3Log<double>((base * so3Exp<double>(plus)).eval())[row] -
                                so3Log<double>((base * so3Exp<double>(minus)).eval())[row]) /
                               (2.0 * kStep);
        max_error = std::max(max_error, std::abs(logged[row].v[k] - numeric));
      }
    }
    expect(max_error < 1e-6, "R·Exp(δ) 经 so3Log 的 Jet Jacobian 与中心差分不符");
    if (max_error >= 1e-6) {
      std::cerr << "  max_error = " << max_error << '\n';
    }
  }

  if (failure_count != 0) {
    std::cerr << "jet smoke test failed with " << failure_count << " error(s)\n";
    return 1;
  }
  std::cout << "jet smoke test passed\n";
  return 0;
}
