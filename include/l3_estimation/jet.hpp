#pragma once

#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <ostream>

// 前向模式自动微分。
//
// 对齐 awakening 用的 ceres::Jet，但不引入 Ceres 依赖：整条 ESEKF 链路只用到
// 十来个算子，自己写一份反而比拖进 ceres/internal/* 干净，也让"误差状态的
// Jacobian 是怎么来的"这件事留在仓库里可读。
//
// 一个 Jet 携带函数值 a 和它对 N 个自变量的偏导 v。把 v 播种成单位阵推一遍
// 计算图，出口处每一行 v 就是该输出对全部输入的梯度——这正是 ESEKF 求
// F = ∂(f(x̌ ⊞ δ) ⊟ f(x̌))/∂δ 的做法，见 docs/esekf_uvl_port.md 第 6.1 节。
namespace L3Estimation {

template <int N>
struct Jet
{
  using Derivative = Eigen::Matrix<double, N, 1>;

  double a{0.0};
  Derivative v{Derivative::Zero()};

  Jet() = default;

  // 常量：导数为零。隐式转换是有意的，这样 T(2.0) * x 之类的写法能直接用。
  Jet(double value) : a(value), v(Derivative::Zero()) {}

  Jet(double value, const Derivative & derivative) : a(value), v(derivative) {}

  // 把第 index 个自变量播种成"对自己求导为 1"。
  static Jet seed(double value, int index)
  {
    Jet result(value);
    result.v[index] = 1.0;
    return result;
  }

  Jet & operator+=(const Jet & rhs)
  {
    a += rhs.a;
    v += rhs.v;
    return *this;
  }

  Jet & operator-=(const Jet & rhs)
  {
    a -= rhs.a;
    v -= rhs.v;
    return *this;
  }

  // 乘积法则：(fg)' = f'g + fg'
  Jet & operator*=(const Jet & rhs)
  {
    v = v * rhs.a + rhs.v * a;
    a *= rhs.a;
    return *this;
  }

  // 商法则：(f/g)' = (f'g - fg') / g²
  Jet & operator/=(const Jet & rhs)
  {
    const double inv = 1.0 / rhs.a;
    v = (v * rhs.a - rhs.v * a) * (inv * inv);
    a *= inv;
    return *this;
  }
};

// --- 一元 ---------------------------------------------------------------

template <int N>
Jet<N> operator+(const Jet<N> & x)
{
  return x;
}

template <int N>
Jet<N> operator-(const Jet<N> & x)
{
  return Jet<N>(-x.a, -x.v);
}

// --- 二元算术 -----------------------------------------------------------
//
// 每个算子都要三个重载：Jet⊕Jet、Jet⊕double、double⊕Jet。少写一个，
// 表达式里出现字面量时就会退化成隐式构造再运算，虽然结果对但多一次拷贝。

template <int N>
Jet<N> operator+(const Jet<N> & f, const Jet<N> & g)
{
  return Jet<N>(f.a + g.a, f.v + g.v);
}

template <int N>
Jet<N> operator+(const Jet<N> & f, double s)
{
  return Jet<N>(f.a + s, f.v);
}

template <int N>
Jet<N> operator+(double s, const Jet<N> & f)
{
  return Jet<N>(s + f.a, f.v);
}

template <int N>
Jet<N> operator-(const Jet<N> & f, const Jet<N> & g)
{
  return Jet<N>(f.a - g.a, f.v - g.v);
}

template <int N>
Jet<N> operator-(const Jet<N> & f, double s)
{
  return Jet<N>(f.a - s, f.v);
}

template <int N>
Jet<N> operator-(double s, const Jet<N> & f)
{
  return Jet<N>(s - f.a, -f.v);
}

template <int N>
Jet<N> operator*(const Jet<N> & f, const Jet<N> & g)
{
  return Jet<N>(f.a * g.a, f.v * g.a + g.v * f.a);
}

template <int N>
Jet<N> operator*(const Jet<N> & f, double s)
{
  return Jet<N>(f.a * s, f.v * s);
}

template <int N>
Jet<N> operator*(double s, const Jet<N> & f)
{
  return Jet<N>(s * f.a, s * f.v);
}

template <int N>
Jet<N> operator/(const Jet<N> & f, const Jet<N> & g)
{
  const double inv = 1.0 / g.a;
  const double value = f.a * inv;
  // (f'g - fg')/g² 写成 (f' - value·g')/g，少一次平方，数值上也更稳。
  return Jet<N>(value, (f.v - value * g.v) * inv);
}

template <int N>
Jet<N> operator/(const Jet<N> & f, double s)
{
  const double inv = 1.0 / s;
  return Jet<N>(f.a * inv, f.v * inv);
}

template <int N>
Jet<N> operator/(double s, const Jet<N> & g)
{
  const double inv = 1.0 / g.a;
  const double value = s * inv;
  return Jet<N>(value, -value * inv * g.v);
}

// --- 比较 ---------------------------------------------------------------
//
// 只比函数值。分支选择不参与求导，这也是为什么 clamp 里的 fmin/fmax 在
// 边界处导数是分段的——那是这套写法固有的近似，不是 bug。

#define L3_JET_COMPARISON(op)                                                       \
  template <int N>                                                                  \
  bool operator op(const Jet<N> & f, const Jet<N> & g) { return f.a op g.a; }        \
  template <int N>                                                                  \
  bool operator op(const Jet<N> & f, double s) { return f.a op s; }                  \
  template <int N>                                                                  \
  bool operator op(double s, const Jet<N> & g) { return s op g.a; }

L3_JET_COMPARISON(<)
L3_JET_COMPARISON(<=)
L3_JET_COMPARISON(>)
L3_JET_COMPARISON(>=)
L3_JET_COMPARISON(==)
L3_JET_COMPARISON(!=)

#undef L3_JET_COMPARISON

// --- 初等函数 -----------------------------------------------------------
//
// 全部放在 L3Estimation 命名空间里，靠 ADL 找到：模板代码写 sqrt(x) 时，
// x 是 double 就走 std::sqrt（配合调用点的 using std::sqrt），是 Jet 就走
// 这里。so3_exp / armor_pose / points_to_observation 依赖这一点。

template <int N>
Jet<N> sqrt(const Jet<N> & f)
{
  const double value = std::sqrt(f.a);
  // d/dx √x = 1/(2√x)。x = 0 处导数无穷，调用方要保证不在零点求导——
  // so3_exp 的小角度分支正是为此存在。
  return Jet<N>(value, f.v * (0.5 / value));
}

template <int N>
Jet<N> exp(const Jet<N> & f)
{
  const double value = std::exp(f.a);
  return Jet<N>(value, f.v * value);
}

template <int N>
Jet<N> log(const Jet<N> & f)
{
  return Jet<N>(std::log(f.a), f.v * (1.0 / f.a));
}

template <int N>
Jet<N> sin(const Jet<N> & f)
{
  return Jet<N>(std::sin(f.a), f.v * std::cos(f.a));
}

template <int N>
Jet<N> cos(const Jet<N> & f)
{
  return Jet<N>(std::cos(f.a), f.v * (-std::sin(f.a)));
}

template <int N>
Jet<N> acos(const Jet<N> & f)
{
  // d/dx acos(x) = -1/√(1-x²)。x → ±1 时发散，so3_log 的 cos_theta > 1-1e-12
  // 早退分支就是挡这个。
  const double d = -1.0 / std::sqrt(1.0 - f.a * f.a);
  return Jet<N>(std::acos(f.a), f.v * d);
}

template <int N>
Jet<N> atan2(const Jet<N> & f, const Jet<N> & g)
{
  // d atan2(f, g) = (f'g - fg') / (f² + g²)
  const double denominator = 1.0 / (f.a * f.a + g.a * g.a);
  return Jet<N>(std::atan2(f.a, g.a), (f.v * g.a - g.v * f.a) * denominator);
}

template <int N>
Jet<N> abs(const Jet<N> & f)
{
  return f.a < 0.0 ? -f : f;
}

template <int N>
Jet<N> floor(const Jet<N> & f)
{
  // 导数恒为零。normalize_angle 依赖这一点：加减 2π 不改变角度的物理含义，
  // 导数当然也不该变。
  return Jet<N>(std::floor(f.a), Jet<N>::Derivative::Zero());
}

template <int N>
Jet<N> fmin(const Jet<N> & f, const Jet<N> & g)
{
  return f.a < g.a ? f : g;
}

template <int N>
Jet<N> fmax(const Jet<N> & f, const Jet<N> & g)
{
  return f.a > g.a ? f : g;
}

template <int N>
bool isfinite(const Jet<N> & f)
{
  return std::isfinite(f.a) && f.v.allFinite();
}

template <int N>
std::ostream & operator<<(std::ostream & os, const Jet<N> & f)
{
  return os << "[" << f.a << " ; " << f.v.transpose() << "]";
}

}  // namespace L3Estimation

// Eigen 需要知道怎么对待这个标量类型，否则 Eigen::Matrix<Jet<N>, 3, 3> 用不了。
// so3_exp / so3_log 全程在 Eigen 矩阵里跑 Jet，这段是必需的。
namespace Eigen {

template <int N>
struct NumTraits<L3Estimation::Jet<N>>
{
  using Real = L3Estimation::Jet<N>;
  using NonInteger = L3Estimation::Jet<N>;
  using Nested = L3Estimation::Jet<N>;
  using Literal = L3Estimation::Jet<N>;

  static L3Estimation::Jet<N> dummy_precision()
  {
    return L3Estimation::Jet<N>(NumTraits<double>::dummy_precision());
  }
  static L3Estimation::Jet<N> epsilon()
  {
    return L3Estimation::Jet<N>(std::numeric_limits<double>::epsilon());
  }
  static L3Estimation::Jet<N> highest()
  {
    return L3Estimation::Jet<N>((std::numeric_limits<double>::max)());
  }
  static L3Estimation::Jet<N> lowest()
  {
    return L3Estimation::Jet<N>(-(std::numeric_limits<double>::max)());
  }
  static int digits10() { return NumTraits<double>::digits10(); }

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    // 一个 Jet 的算术代价约等于 (N + 1) 个 double。
    ReadCost = 1,
    AddCost = 1 + N,
    MulCost = 3 + 2 * N
  };
};

}  // namespace Eigen
