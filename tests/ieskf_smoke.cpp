// IteratedKalmanFilter 的行为冒烟测试。不依赖相机、串口和推理后端，
// 因此在 xmake.lua 里直接列源文件而不是 add_deps("newvision")。

#include "l3_estimation/ieskf.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

namespace {

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << "ieskf smoke test failed: " << message << '\n';
    std::exit(1);
  }
}

// 把二维状态 [x, y] 映射到 [方位角, 距离]。这是 target_estimator 里
// xyz2ypd 的低维等价物，非线性强度相当。
Eigen::VectorXd polarObservation(const Eigen::VectorXd & x)
{
  Eigen::VectorXd z(2);
  z << std::atan2(x[1], x[0]), std::hypot(x[0], x[1]);
  return z;
}

Eigen::MatrixXd polarJacobian(const Eigen::VectorXd & x)
{
  const double r2 = x[0] * x[0] + x[1] * x[1];
  const double r = std::sqrt(r2);
  Eigen::MatrixXd H(2, 2);
  H << -x[1] / r2, x[0] / r2,
        x[0] / r,  x[1] / r;
  return H;
}

// 与 TrackedTarget 一致的流形运算：下标 1 当作周期角度处理。
Eigen::VectorXd wrapAdd(const Eigen::VectorXd & a, const Eigen::VectorXd & b)
{
  Eigen::VectorXd result = a + b;
  result[1] = std::remainder(result[1], 2.0 * std::numbers::pi);
  return result;
}

Eigen::VectorXd wrapMinus(const Eigen::VectorXd & a, const Eigen::VectorXd & b)
{
  Eigen::VectorXd result = a - b;
  result[1] = std::remainder(result[1], 2.0 * std::numbers::pi);
  return result;
}

// max_iterations 为 1 时必须与基类 update() 逐位一致。这是回归保护：
// 迭代路径不能悄悄改变既有的非迭代行为。
void testSingleIterationMatchesEkf()
{
  Eigen::VectorXd x0(2);
  x0 << 3.0, 1.0;
  const Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(2, 2) * 0.5;
  const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * 1e-2;

  Eigen::VectorXd z(2);
  z << std::atan2(1.4, 3.3), std::hypot(3.3, 1.4);

  // 基类在先验点线性化一次，H 是矩阵。
  L3Estimation::ExtendedKalmanFilter ekf(x0, P0);
  const Eigen::MatrixXd H_prior = polarJacobian(x0);
  ekf.update(z, H_prior, R, polarObservation);

  // 迭代类限制成 1 次迭代，H 是函数但只在先验点求值一次。
  L3Estimation::IteratedKalmanFilter iekf(x0, P0);
  iekf.update(z, polarJacobian, R, polarObservation,
              [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
                return a - b;
              },
              1);

  require((ekf.x - iekf.x).norm() == 0.0, "single iteration state must match EKF bit-for-bit");
  require((ekf.P - iekf.P).norm() == 0.0, "single iteration covariance must match EKF bit-for-bit");
  require(iekf.lastIterations() == 1, "lastIterations must report 1");
  require(ekf.last_nis == iekf.last_nis, "NIS must match the non-iterated path");
  std::cout << "  [ok] max_iterations=1 matches ExtendedKalmanFilter\n";
}

// 先验离真值较远时，迭代重线性化应当把后验推得更接近观测。
void testIterationReducesResidual()
{
  const Eigen::VectorXd truth = (Eigen::VectorXd(2) << 4.0, 3.0).finished();
  const Eigen::VectorXd z = polarObservation(truth);

  // 故意给一个偏差很大的先验，让先验点的 Jacobian 明显失真。
  Eigen::VectorXd x0(2);
  x0 << 1.5, 5.5;
  const Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(2, 2) * 4.0;
  const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * 1e-4;

  L3Estimation::IteratedKalmanFilter single(x0, P0);
  single.update(z, polarJacobian, R, polarObservation,
                [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; },
                1);

  L3Estimation::IteratedKalmanFilter iterated(x0, P0);
  iterated.update(z, polarJacobian, R, polarObservation,
                  [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; },
                  5, 1e-8);

  const double single_residual = (z - polarObservation(single.x)).norm();
  const double iterated_residual = (z - polarObservation(iterated.x)).norm();
  const double single_error = (truth - single.x).norm();
  const double iterated_error = (truth - iterated.x).norm();

  std::cout << "  single: residual=" << single_residual
            << " error=" << single_error << '\n';
  std::cout << "  iterated(" << iterated.lastIterations()
            << "): residual=" << iterated_residual
            << " error=" << iterated_error << '\n';

  require(iterated.lastIterations() > 1, "iteration must actually run");
  require(iterated_residual < single_residual,
          "iterated update must shrink the measurement residual");
  require(iterated_error < single_error,
          "iterated update must land closer to ground truth");
  std::cout << "  [ok] iteration reduces residual and state error\n";
}

// yaw 跨越 ±π 时，x_minus 必须走最短圆周差，否则先验残差会带上 2π。
void testAngleWrapAroundPi()
{
  Eigen::VectorXd x0(2);
  x0 << 2.0, std::numbers::pi - 0.02;
  const Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(2, 2) * 0.3;
  const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * 1e-3;

  // 观测直接落在 -π 一侧，与先验只差 0.04 rad。
  auto identity_h = [](const Eigen::VectorXd & x) { return x; };
  auto identity_H = [](const Eigen::VectorXd &) {
    return Eigen::MatrixXd(Eigen::MatrixXd::Identity(2, 2));
  };
  auto wrap_z = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    Eigen::VectorXd result = a - b;
    result[1] = std::remainder(result[1], 2.0 * std::numbers::pi);
    return result;
  };

  Eigen::VectorXd z(2);
  z << 2.0, -std::numbers::pi + 0.02;

  L3Estimation::IteratedKalmanFilter iekf(x0, P0, wrapAdd, wrapMinus);
  iekf.update(z, identity_H, R, identity_h, wrap_z, 5, 1e-10);

  // 后验角度必须停在 ±π 附近，而不是被 2π 伪残差拉到 0 附近。
  const double distance_to_pi =
    std::abs(std::abs(iekf.x[1]) - std::numbers::pi);
  require(iekf.x.allFinite(), "state must stay finite across the wrap");
  require(distance_to_pi < 0.05, "posterior angle must stay near the +-pi seam");
  std::cout << "  [ok] angle wrap handled, posterior=" << iekf.x[1] << '\n';
}

// 观测里出现非有限值时必须原样退回先验，绝不污染状态和协方差。
void testNonFiniteMeasurementIsRejected()
{
  Eigen::VectorXd x0(2);
  x0 << 3.0, 1.0;
  const Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(2, 2) * 0.5;
  const Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * 1e-2;

  L3Estimation::IteratedKalmanFilter iekf(x0, P0);
  Eigen::VectorXd z(2);
  z << std::numeric_limits<double>::quiet_NaN(), 3.2;
  iekf.update(z, polarJacobian, R, polarObservation);

  require(iekf.x.allFinite(), "state must stay finite after a NaN measurement");
  require(iekf.P.allFinite(), "covariance must stay finite after a NaN measurement");
  require((iekf.x - x0).norm() == 0.0, "state must be unchanged");
  require(iekf.recent_nis_failures.back() == 1, "NaN NIS must count as a failure");
  std::cout << "  [ok] non-finite measurement rejected\n";
}

}  // namespace

int main()
{
  testSingleIterationMatchesEkf();
  testIterationReducesResidual();
  testAngleWrapAroundPi();
  testNonFiniteMeasurementIsRejected();
  std::cout << "ieskf smoke test passed\n";
  return 0;
}
