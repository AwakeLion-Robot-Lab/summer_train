#pragma once

#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/so3.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <numbers>

// 整车模型：十三维状态、由状态生成每块装甲板位姿的结构先验、流形上的 ⊞/⊟，
// 以及恒速度运动模型。对齐 awakening 的 armor_track/motion_model.hpp。
//
// 与旧的 TrackedTarget 相比，本质区别只有一个：姿态是完整的 SO(3) 旋转向量，
// 不是单个 yaw 标量。这带来两个连锁后果——状态不再住在向量空间（于是需要
// ⊞/⊟ 和误差状态），而过程噪声可以在车体系表达（于是"地面车不会突然上下
// 加速"这个先验才有地方写）。推导见 docs/esekf_uvl_port.md 第 2、3 节。
namespace L3Estimation::VehicleModel {

// 状态下标。
//
// 前九维（0..8）**不可改动**：L4 的 Planner 按下标直接读 x[0]、x[2]（车心
// 平面位置）、x[7]（角速度）、x[8]（第一组半径），见 planner.cpp:64/84/240。
// 这也正好与 awakening 的布局逐位重合，移植时不需要任何转换。
//
// 代价是旋转三维不连续：z 在 6，y 和 x 挤到 11、12。看着别扭，但把它们改成
// 连续就会撞上 VYAW=7 与 LOG_R1=8，直接破坏 L4 契约。所以这个不连续是
// 载荷，不是历史遗留。
namespace idx
{
enum {
  CX, VCX,     // 车心 x 与其速度
  CY, VCY,     // 车心 y 与其速度
  CZ, VCZ,     // 车心 z 与其速度
  ROT_Z,       // 整车姿态旋转向量的 z 分量
  VYAW,        // 绕**车体** z 轴的角速度
  LOG_R1,      // 第一组装甲板半径的对数
  P1,          // 复用槽位，含义随目标类型变
  P2,          // 同上
  ROT_Y,       // 整车姿态旋转向量的 y 分量
  ROT_X,       // 整车姿态旋转向量的 x 分量
  kStateSize
};

// P1 / P2 是"变形金刚"：同一个 double，四板车和前哨站上含义完全不同。
// 所有碰这两个槽位的代码都必须先分支判 ArmorName，否则就是静默算错。
constexpr int LOG_R2 = P1;       // 四板车：第二组半径的对数
constexpr int HEIGHT = P2;       // 四板车：奇偶板的高度差
constexpr int OUTPOST_DZ1 = P1;  // 前哨站：1 号板相对 0 号板的高度
constexpr int OUTPOST_DZ2 = P2;  // 前哨站：2 号板相对 0 号板的高度
}  // namespace idx

constexpr int kStateSize = idx::kStateSize;
static_assert(kStateSize == 13, "L4 与遥测按十三维读状态");

// 旋转三维的下标，按 (x, y, z) 顺序，供 ⊞/⊟ 使用。
constexpr int kRotationIndex[3] = {idx::ROT_X, idx::ROT_Y, idx::ROT_Z};

// 半径的物理范围，用于防发散。这类物理常量留在代码里，不出到配置。
constexpr double kMinArmorRadius = 0.05;
constexpr double kMaxArmorRadius = 1.0;
// 前哨站半径与转速由规则固定，不参与估计。
constexpr double kOutpostRadius = 0.55 / 2.0;
constexpr double kOutpostYawRate = 2.51;
// 高度差与角速度的发散判据。超界不是"估偏了一点"，而是关联错了，所以复位。
constexpr double kMaxHeightOffset = 0.5;
constexpr double kMaxOutpostHeightOffset = 0.3;
constexpr double kMaxYawRate = 20.0;

// 是否估计完整三自由度姿态。关掉则退化为只估 yaw，等价于旧的整车 EKF——
// 留这个开关是为了能在同一段回放上做 A/B，隔离"误差状态"与"图像观测"
// 各自贡献了多少。
inline constexpr bool kEstimateFullAttitude = true;

// 折回 (-π, π]。无分支写法，对 Jet 友好：floor 的导数恒为零，而加减 2π 不
// 改变角度的物理含义，导数当然也不该变。
template <typename T>
T normalizeAngle(T angle)
{
  using std::floor;
  const T two_pi = T(2.0 * std::numbers::pi);
  return angle - two_pi * floor((angle + T(std::numbers::pi)) / two_pi);
}

// Rz(yaw) · Ry(pitch)。装甲板在车体系的姿态只用得到这两轴。
template <typename T>
Eigen::Matrix<T, 3, 3> rotationZY(const T & yaw, const T & pitch)
{
  using std::sin;
  using std::cos;
  const T cos_yaw = cos(yaw);
  const T sin_yaw = sin(yaw);
  const T cos_pitch = cos(pitch);
  const T sin_pitch = sin(pitch);

  Eigen::Matrix<T, 3, 3> result;
  result << cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
            sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch,
            -sin_pitch, T(0.0), cos_pitch;
  return result;
}

// 目标是否退化为"只绕竖直轴转"。前哨站是固定装置绕竖直轴匀速转，基地不转，
// 给它们估 roll/pitch 只会引入不可观测自由度让滤波器漂。
constexpr bool yawOnlyTarget(ArmorName name) noexcept
{
  return name == ArmorName::Outpost || name == ArmorName::BaseSmall ||
         name == ArmorName::BaseLarge || !kEstimateFullAttitude;
}

constexpr bool isBase(ArmorName name) noexcept
{
  return name == ArmorName::BaseSmall || name == ArmorName::BaseLarge;
}

// 整车姿态。
template <typename T>
Eigen::Matrix<T, 3, 3> vehicleRotation(const T * x, ArmorName name)
{
  if (yawOnlyTarget(name)) {
    return so3Exp<T>(Eigen::Matrix<T, 3, 1>(T(0.0), T(0.0), x[idx::ROT_Z]));
  }
  return so3Exp<T>(Eigen::Matrix<T, 3, 1>(x[idx::ROT_X], x[idx::ROT_Y], x[idx::ROT_Z]));
}

// 第 id 块板用哪一组半径。
//
// is_r2 的两个条件缺一不可：三板车（前哨、基地）所有板到轴心的距离物理上
// 相同，共用第一组；四板车前后一组、左右一组，底盘不是正方形。
//
// exp(log_r) 是半径保正的全部机制——状态里存对数，取出来必然为正，不需要
// 任何约束或投影。
template <typename T>
T armorRadius(const T * x, int id, int armor_num, ArmorName name)
{
  using std::exp;
  if (isBase(name)) {
    return T(0.0);  // 基地的板就在中心，不绕转
  }
  const bool is_r2 = (armor_num == 4) && ((id & 1) != 0);
  return exp(is_r2 ? x[idx::LOG_R2] : x[idx::LOG_R1]);
}

// 整车在世界系的位姿：状态的前六维取位置，姿态三维取朝向。
template <typename T>
Eigen::Transform<T, 3, Eigen::Isometry> vehiclePose(const T * x, ArmorName name)
{
  Eigen::Transform<T, 3, Eigen::Isometry> pose =
    Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
  pose.translation() << x[idx::CX], x[idx::CY], x[idx::CZ];
  pose.linear() = vehicleRotation<T>(x, name);
  return pose;
}

// 第 id 块装甲板在世界系的位姿。**整车估计思路的全部内容。**
//
// 板的方位角是常量而非待估量：θ_i = 2πi/N。半径和高度差是状态量。于是
// "装甲板跳变"被转化为确定的刚体几何关系——无论当前看到的是正面板、侧面板
// 还是相邻板的一条灯条，它们都只是同一个十三维状态在不同 i 上的投影。
//
// 注意板心位置取 -r·(cosθ, sinθ) 而姿态绕 z 转 +θ，两者反向，所以**板的
// x 轴指向车心**，可见面是它的 -x 侧。判可见性时要用 -axis_x，搞反会把
// 背面的板当成正对的。
template <typename T>
Eigen::Transform<T, 3, Eigen::Isometry> armorPose(
  const T * x, int id, int armor_num, ArmorName name)
{
  const T yaw = normalizeAngle(T(id) * T(2.0 * std::numbers::pi / armor_num));
  const T radius = armorRadius<T>(x, id, armor_num, name);

  using std::sin;
  using std::cos;
  const T offset_x = -cos(yaw) * radius;
  const T offset_y = -sin(yaw) * radius;

  T offset_z = T(0.0);
  if (name == ArmorName::Outpost) {
    if (id == 1) {
      offset_z = x[idx::OUTPOST_DZ1];
    } else if (id == 2) {
      offset_z = x[idx::OUTPOST_DZ2];
    }
  } else if ((armor_num == 4) && ((id & 1) != 0)) {
    offset_z = x[idx::HEIGHT];
  }

  Eigen::Transform<T, 3, Eigen::Isometry> armor_in_vehicle =
    Eigen::Transform<T, 3, Eigen::Isometry>::Identity();
  armor_in_vehicle.translation() << offset_x, offset_y, offset_z;
  // 板面后仰角与 PnP 物点、叠加层法向箭头、火控命中窗口共用同一份来源。
  armor_in_vehicle.linear() = rotationZY<T>(yaw, T(armorPitchOf(name)));

  return vehiclePose<T>(x, name) * armor_in_vehicle;
}

// --- 流形运算 -----------------------------------------------------------
//
// 普通 EKF 的三个动作（x += K·r、P = FPFᵀ+Q、P 是状态减均值的二阶矩）都假设
// 状态住在向量空间。旋转不住在向量空间。解法是把状态劈成"名义状态在流形上"
// 与"误差状态在切空间里"，滤波器的 P 是**误差状态**的协方差。
//
// 扰动取右乘（局部/体坐标系）而非左乘，是被下游逼的：过程噪声要在车体系表达
// （地面车不会突然上下加速），而 vyaw 是绕车体 z 轴的角速度。选左乘的话
// process_noise 里 Q 的旋转和 ROT_Z–VYAW 耦合块都得重写。

template <class StateVector>
auto stateRotation(const StateVector & state)
{
  using Scalar = typename std::decay_t<StateVector>::Scalar;
  return so3Exp<Scalar>(Eigen::Matrix<Scalar, 3, 1>(
    state[idx::ROT_X], state[idx::ROT_Y], state[idx::ROT_Z]));
}

// x̌ ⊞ δ：欧氏分量直接加，姿态右乘注入 R ← R·Exp(δ_rot)。
//
// 循环里**跳过**旋转三维、循环外再单独处理，不是可以省的写法：
// stateRotation(nominal) 在循环之后被调用，读到的必须是未被误加的旋转。
template <class DeltaVector, class StateVector>
void injectState(const DeltaVector & delta, StateVector & nominal)
{
  using Scalar = typename std::decay_t<DeltaVector>::Scalar;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  for (int i = 0; i < kStateSize; ++i) {
    const bool is_rotation =
      i == idx::ROT_X || i == idx::ROT_Y || i == idx::ROT_Z;
    if (!is_rotation) {
      nominal[i] += delta[i];
    }
  }

  const Vector3 delta_rotation(delta[idx::ROT_X], delta[idx::ROT_Y], delta[idx::ROT_Z]);
  const Vector3 injected =
    so3Log<Scalar>((stateRotation(nominal) * so3Exp<Scalar>(delta_rotation)).eval());
  nominal[idx::ROT_X] = injected.x();
  nominal[idx::ROT_Y] = injected.y();
  nominal[idx::ROT_Z] = injected.z();
}

// x ⊟ x̌：欧氏分量直接减，姿态取 Log(Řᵀ·R)。injectState 的逆。
//
// 转置的位置必须与右乘配套。写成 Log(R·Řᵀ)（左乘形式）互逆性就破了，而且
// **不会报错**——滤波器照样跑，只是 Jacobian 全错、协方差没有意义、遇到大
// 机动就发散。tests/vehicle_model_smoke.cpp 的第一条断言守的就是这个。
template <class StateVector, class DeltaVector>
void boxMinusState(const StateVector & nominal, const StateVector & value, DeltaVector & delta)
{
  using Scalar = typename std::decay_t<StateVector>::Scalar;
  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;

  delta = value - nominal;  // 旋转三维此刻是错的，下面覆盖掉

  const Vector3 delta_rotation =
    so3Log<Scalar>((stateRotation(nominal).transpose() * stateRotation(value)).eval());
  delta[idx::ROT_X] = delta_rotation.x();
  delta[idx::ROT_Y] = delta_rotation.y();
  delta[idx::ROT_Z] = delta_rotation.z();
}

// --- 运动模型 -----------------------------------------------------------

// 恒速度平移 + 绕车体 z 轴恒角速度，其余状态随机游走。模型朴素是有意的：
// 复杂度在观测端，不在这里。
struct Motion
{
  double dt{0.0};
  ArmorName name{ArmorName::Unknown};
  // 前哨站转向判明后角速度钉死为规则常量。0 表示尚未判明，按状态量推进。
  int outpost_direction{0};

  template <typename T>
  void operator()(const T * x0, T * x1) const
  {
    std::copy(x0, x0 + kStateSize, x1);  // 默认全部不变：半径、高度都是随机游走

    x1[idx::CX] += x0[idx::VCX] * T(dt);
    x1[idx::CY] += x0[idx::VCY] * T(dt);
    x1[idx::CZ] += x0[idx::VCZ] * T(dt);

    if (!isBase(name)) {
      Eigen::Matrix<T, 3, 1> delta_rotation;
      if (name == ArmorName::Outpost && outpost_direction != 0) {
        const T rate = T(outpost_direction > 0 ? kOutpostYawRate : -kOutpostYawRate);
        delta_rotation << T(0.0), T(0.0), rate * T(dt);
        x1[idx::VYAW] = rate;
      } else {
        delta_rotation << T(0.0), T(0.0), x0[idx::VYAW] * T(dt);
      }

      // 右乘：delta_rotation 只有 z 分量，表达的是"绕车体自身竖直轴"。
      // 若改左乘，这就变成"绕世界 z 轴"，车一有 roll/pitch 就错。
      const Eigen::Matrix<T, 3, 3> rotated =
        (so3Exp<T>(Eigen::Matrix<T, 3, 1>(x0[idx::ROT_X], x0[idx::ROT_Y], x0[idx::ROT_Z])) *
         so3Exp<T>(delta_rotation))
          .eval();
      const Eigen::Matrix<T, 3, 1> updated = so3Log<T>(rotated);
      x1[idx::ROT_X] = updated.x();
      x1[idx::ROT_Y] = updated.y();
      x1[idx::ROT_Z] = updated.z();
    }

    clamp(x1);
  }

  // 防发散的硬约束。半径用饱和（估偏了一点，拉回边界合理），高度差和角速度
  // 用归零（超界几乎一定是关联错误，等于承认这一维已经没救、重来）。
  //
  // 它在状态转移内部，所以 Jet 会穿过这些分支。fmin/fmax 在边界处导数分段，
  // 理论上让 F 在饱和瞬间不连续；实践中很少触发，是个已知的近似。
  template <typename T>
  void clamp(T * x) const
  {
    using std::fmin;
    using std::fmax;
    using std::abs;

    x[idx::LOG_R1] = fmax(
      T(std::log(kMinArmorRadius)), fmin(T(std::log(kMaxArmorRadius)), x[idx::LOG_R1]));

    if (name == ArmorName::Outpost) {
      if (abs(x[idx::OUTPOST_DZ1]) > T(kMaxOutpostHeightOffset)) {
        x[idx::OUTPOST_DZ1] = T(0.0);
      }
      if (abs(x[idx::OUTPOST_DZ2]) > T(kMaxOutpostHeightOffset)) {
        x[idx::OUTPOST_DZ2] = T(0.0);
      }
      x[idx::LOG_R1] = T(std::log(kOutpostRadius));  // 规则固定，不估计
    } else {
      x[idx::LOG_R2] = fmax(
        T(std::log(kMinArmorRadius)), fmin(T(std::log(kMaxArmorRadius)), x[idx::LOG_R2]));
      if (abs(x[idx::HEIGHT]) > T(kMaxHeightOffset)) {
        x[idx::HEIGHT] = T(0.0);
      }
    }

    if (abs(x[idx::VYAW]) > T(kMaxYawRate)) {
      x[idx::VYAW] = T(0.0);
    }
    if (isBase(name)) {
      x[idx::VYAW] = T(0.0);
    }
  }
};

}  // namespace L3Estimation::VehicleModel
