#pragma once

#include "l3_estimation/jet.hpp"

#include <Eigen/Core>

#include <cmath>

// SO(3) 的指数与对数映射，模板化在标量类型上，double 和 Jet 都能跑。
//
// 误差状态滤波把姿态的所有运算都关进"小角度"这个安全区：δ 恒在零附近，
// 所以 so3Exp 的泰勒分支才是常走的路径，而 so3Log 在 θ→π 处的奇异永远
// 碰不到。推导见 docs/esekf_uvl_port.md 第 1 节。
namespace L3Estimation {

// 反对称矩阵（hat 算子），把叉乘写成矩阵乘法：so3Hat(w) * v == w × v。
template <typename T>
Eigen::Matrix<T, 3, 3> so3Hat(const Eigen::Matrix<T, 3, 1> & w)
{
  Eigen::Matrix<T, 3, 3> result;
  result << T(0.0), -w.z(), w.y(),
            w.z(), T(0.0), -w.x(),
            -w.y(), w.x(), T(0.0);
  return result;
}

// 指数映射：旋转向量 → 旋转矩阵（罗德里格斯公式）。
//
//   R = I + A·φ^ + B·(φ^)²,   A = sin θ / θ,   B = (1 - cos θ) / θ²
//
// θ → 0 时 A、B 都是 0/0。极限存在（1 和 1/2），但浮点直接算得到 NaN，所以
// 小角度必须走泰勒展开——这条分支在滤波器里是常态而非边界情况。
template <typename T>
Eigen::Matrix<T, 3, 3> so3Exp(const Eigen::Matrix<T, 3, 1> & phi)
{
  using std::sqrt;
  using std::sin;
  using std::cos;

  const T theta_squared = phi.squaredNorm();

  const Eigen::Matrix<T, 3, 3> hat = so3Hat<T>(phi);
  const Eigen::Matrix<T, 3, 3> hat_squared = hat * hat;

  T a;
  T b;
  if (theta_squared < T(1e-12)) {
    const T theta_fourth = theta_squared * theta_squared;
    a = T(1.0) - theta_squared / T(6.0) + theta_fourth / T(120.0);
    b = T(0.5) - theta_squared / T(24.0) + theta_fourth / T(720.0);
  } else {
    const T theta = sqrt(theta_squared);
    a = sin(theta) / theta;
    b = (T(1.0) - cos(theta)) / theta_squared;
  }

  Eigen::Matrix<T, 3, 3> result = Eigen::Matrix<T, 3, 3>::Identity();
  result += a * hat + b * hat_squared;
  return result;
}

// 对数映射：旋转矩阵 → 旋转向量，so3Exp 的逆。
//
//   cos θ = (tr(R) - 1) / 2            由迹取转角
//   R - Rᵀ = 2 sin θ · a^              由反对称部分取转轴
//   φ = θ / (2 sin θ) · vee(R - Rᵀ)
//
// θ → π 时 sin θ → 0，scale 会发散，这里**没有**保护。调用点只有 ⊞ 和 ⊟，
// 两者的输入都保证接近单位阵，碰不到 π。若在别处复用要自己补。
template <typename T>
Eigen::Matrix<T, 3, 1> so3Log(const Eigen::Matrix<T, 3, 3> & rotation)
{
  using std::sin;
  using std::acos;

  const T cos_theta = (rotation.trace() - T(1.0)) * T(0.5);

  Eigen::Matrix<T, 3, 1> vee;
  vee << rotation(2, 1) - rotation(1, 2),
         rotation(0, 2) - rotation(2, 0),
         rotation(1, 0) - rotation(0, 1);

  // θ → 0 时 θ/(2 sin θ) → 1/2。
  if (cos_theta > T(1.0 - 1e-12)) {
    return T(0.5) * vee;
  }

  const T theta = acos(cos_theta);
  const T scale = theta / (T(2.0) * sin(theta));
  return scale * vee;
}

}  // namespace L3Estimation
