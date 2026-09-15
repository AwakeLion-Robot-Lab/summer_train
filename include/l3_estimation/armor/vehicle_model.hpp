#pragma once

#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/types.hpp"
#include "l3_estimation/so3.hpp"

#include <ceres/jet.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <numbers>
#include <array>

// 整车模型：十三维状态、由状态生成每块装甲板位姿的结构先验、流形上的 ⊞/⊟，
// 以及恒速度运动模型。对齐 awakening 的 armor_track/motion_model.hpp。
//
// 与 sp_vision 只估一个 yaw 标量的整车模型相比，本质区别只有一个：姿态是完整的
// SO(3) 旋转向量。这带来两个连锁后果——状态不再住在向量空间（于是需要
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
constexpr double kMaxArmorRadius = 0.8;
// 前哨站半径与转速由规则固定，不参与估计。
constexpr double kOutpostRadius = 0.55 / 2.0;
constexpr double kOutpostYawRate = 2.51;
// 高度差与角速度的发散判据。超界不是"估偏了一点"，而是关联错了，所以复位。
constexpr double kMaxHeightOffset = 0.5;
constexpr double kMaxOutpostHeightOffset = 0.3;
constexpr double kMaxYawRate = 20.0;

// 折回 (-π, π]。无分支写法，对 Jet 友好：floor 的导数恒为零，而加减 2π 不
// 改变角度的物理含义，导数当然也不该变。
template <typename T>
T normalizeAngle(T angle)
{
  const T two_pi = T(2.0 * std::numbers::pi);
  return angle - two_pi * ceres::floor((angle + T(std::numbers::pi)) / two_pi);
}

// Rz(yaw) · Ry(pitch)。装甲板在车体系的姿态只用得到这两轴。
template <typename T>
Eigen::Matrix<T, 3, 3> rotationZY(const T & yaw, const T & pitch)
{
  const T cos_yaw = ceres::cos(yaw);
  const T sin_yaw = ceres::sin(yaw);
  const T cos_pitch = ceres::cos(pitch);
  const T sin_pitch = ceres::sin(pitch);

  Eigen::Matrix<T, 3, 3> result;
  result << cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
            sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch,
            -sin_pitch, T(0.0), cos_pitch;
  return result;
}

constexpr bool isBase(ArmorName name) noexcept
{
  return name == ArmorName::BaseSmall || name == ArmorName::BaseLarge;
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
  if (isBase(name)) {
    return T(0.0);  // 基地的板就在中心，不绕转
  }
  const bool is_r2 = (armor_num == 4) && ((id & 1) != 0);
  return ceres::exp(is_r2 ? x[idx::LOG_R2] : x[idx::LOG_R1]);
}

// 目标是否退化为"只绕竖直轴转"。前哨站是固定装置绕竖直轴匀速转，基地不转，
// 给它们估 roll/pitch 只会引入不可观测自由度让滤波器漂。
constexpr bool yawOnlyTarget(ArmorName name) noexcept
{
  return name == ArmorName::Outpost || name == ArmorName::BaseSmall ||
         name == ArmorName::BaseLarge;
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

  const T offset_x = -ceres::cos(yaw) * radius;
  const T offset_y = -ceres::sin(yaw) * radius;

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

// --- 前哨站转向投票 -----------------------------------------------------

// 前哨站的转速由规则固定（kOutpostYawRate），未知的只是**转向**。与其把它
// 当自由度交给滤波器估，不如攒够证据判明方向后直接钉死——少估一个自由度，
// 精度和收敛速度都受益。
//
// 照搬 awakening 的 Voter：开机 1 秒内不投票（等状态先收敛），此后每次 yaw
// 变化超过 0.05 rad 就给计数器 ±1，累计绝对值超过 10 票才判定。门限取得高是
// 因为判错方向比判不出更糟——判不出只是退回按状态量推进，判错会让预测朝反
// 方向跑。
struct Voter
{
  enum class Direction
  {
    Collecting,      // 证据不足，仍按状态量推进
    Clockwise,       // +kOutpostYawRate
    Counterclockwise // -kOutpostYawRate
  };

  Direction direction{Direction::Collecting};
  TimePoint start{};
  int clockwise_count{0};
  double last_yaw{0.0};
  bool has_last_yaw{false};

  void reset(TimePoint t) noexcept
  {
    direction = Direction::Collecting;
    start = t;
    clockwise_count = 0;
    last_yaw = 0.0;
    has_last_yaw = false;
  }

  void update(double yaw, TimePoint t) noexcept
  {
    if (t - start < std::chrono::milliseconds(1000)) {
      return;
    }
    if (!has_last_yaw) {
      last_yaw = yaw;
      has_last_yaw = true;
      return;
    }

    const double diff = normalizeAngle(yaw - last_yaw);
    // 太小的变化多半是噪声，不投票，也不更新参考角——否则噪声会把参考角
    // 一点点推着走，永远攒不出证据。
    if (std::abs(diff) < 0.05) {
      return;
    }

    clockwise_count += (diff > 0.0) ? 1 : -1;
    last_yaw = yaw;

    if (std::abs(clockwise_count) > 10) {
      direction = clockwise_count > 0 ? Direction::Clockwise : Direction::Counterclockwise;
    } else {
      direction = Direction::Collecting;
    }
  }

  // 喂给 Motion 的符号：0 表示尚未判明。
  int sign() const noexcept
  {
    switch (direction) {
      case Direction::Clockwise:
        return 1;
      case Direction::Counterclockwise:
        return -1;
      case Direction::Collecting:
        break;
    }
    return 0;
  }
};

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
    x[idx::LOG_R1] = ceres::fmax(
      T(std::log(kMinArmorRadius)), ceres::fmin(T(std::log(kMaxArmorRadius)), x[idx::LOG_R1]));

    if (name == ArmorName::Outpost) {
      if (ceres::abs(x[idx::OUTPOST_DZ1]) > T(kMaxOutpostHeightOffset)) {
        x[idx::OUTPOST_DZ1] = T(0.0);
      }
      if (ceres::abs(x[idx::OUTPOST_DZ2]) > T(kMaxOutpostHeightOffset)) {
        x[idx::OUTPOST_DZ2] = T(0.0);
      }
      x[idx::LOG_R1] = T(std::log(kOutpostRadius));  // 规则固定，不估计
    } else {
      x[idx::LOG_R2] = ceres::fmax(
        T(std::log(kMinArmorRadius)), ceres::fmin(T(std::log(kMaxArmorRadius)), x[idx::LOG_R2]));
      if (ceres::abs(x[idx::HEIGHT]) > T(kMaxHeightOffset)) {
        x[idx::HEIGHT] = T(0.0);
      }
    }

    if (ceres::abs(x[idx::VYAW]) > T(kMaxYawRate)) {
      x[idx::VYAW] = T(0.0);
    }
    if (isBase(name)) {
      x[idx::VYAW] = T(0.0);
    }
  }
};


// --- 过程噪声 -----------------------------------------------------------

// 整车 ESEKF 的过程噪声强度。这些是靠回放标定的主要旋钮，所以出到配置；
// 半径物理范围、前哨固定转速那类物理常量仍留在代码里。
struct NoiseConfig
{
  // 车体系加速度方差 [前, 左, 上]。三个数不相等是这套写法的全部意义所在：
  // 地面轮式车可以突然前后左右加速，但不会突然上下加速。这句话只有在**车体
  // 坐标系**里才成立，所以 Q 必须先在体系建好再旋到世界系。
  Eigen::Vector3d body_acceleration{30.0, 30.0, 1.0};
  // 绕车体 z 轴的角加速度方差。
  double yaw_acceleration{30.0};

  // 前哨站转速由规则固定、轨迹规整，过程噪声显著更小。
  Eigen::Vector3d outpost_body_acceleration{1.0, 1.0, 1.0};
  double outpost_yaw_acceleration{0.01};

  // 半径与高度差的随机游走强度。
  double radius{1e-7};
  double height{1e-7};
  double outpost_height{1e-7};
  // 非 yaw 姿态漂移，吸收车体 roll/pitch 小幅误差、地面坡度和外参残差。
  double roll_pitch{0.1};
};

// 构造过程噪声矩阵。
//
// 平移与 yaw 都用常加速度模型：把未建模的加速度当白噪声 a ~ N(0, σ²)，在 dt
// 内它对位置和速度的影响是 G = [½dt², dt]ᵀ，于是那个 2×2 块是
// G σ² Gᵀ = σ² [[¼dt⁴, ½dt³], [½dt³, dt²]]。四个系数就是这么来的。
//
// 平移部分先在车体系建对角阵再旋到世界系（协方差的标准传播律
// Cov(Ra) = R Cov(a) Rᵀ）；yaw 部分**不需要旋转**，因为选了右乘之后误差状态
// 本来就定义在体系里。若当初选左乘，这里既要把 yaw 噪声旋进世界系，
// ROT_Z–VYAW 的耦合块还会变成三维稠密的——右乘的选择让这段保持简单。
inline Eigen::Matrix<double, kStateSize, kStateSize> processNoise(
  const Eigen::VectorXd & x, double dt, ArmorName name, const NoiseConfig & config)
{
  const bool outpost = name == ArmorName::Outpost;
  const Eigen::Vector3d body_acceleration =
    outpost ? config.outpost_body_acceleration : config.body_acceleration;
  const double yaw_acceleration =
    outpost ? config.outpost_yaw_acceleration : config.yaw_acceleration;

  Eigen::Matrix<double, kStateSize, kStateSize> q;
  q.setZero();

  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;

  // ① 平移：车体系加速度噪声旋到世界系
  const Eigen::Matrix3d vehicle_rotation = vehicleRotation<double>(x.data(), name);
  const Eigen::Matrix3d acceleration_in_world =
    vehicle_rotation * body_acceleration.asDiagonal() * vehicle_rotation.transpose();

  constexpr std::array<int, 3> position_index{idx::CX, idx::CY, idx::CZ};
  constexpr std::array<int, 3> velocity_index{idx::VCX, idx::VCY, idx::VCZ};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double value = acceleration_in_world(i, j);
      q(position_index[i], position_index[j]) = 0.25 * dt4 * value;
      q(position_index[i], velocity_index[j]) = 0.5 * dt3 * value;
      q(velocity_index[i], position_index[j]) = 0.5 * dt3 * value;
      q(velocity_index[i], velocity_index[j]) = dt2 * value;
    }
  }

  // ② yaw：常角加速度，误差已在体系，不旋转
  q(idx::VYAW, idx::VYAW) += dt2 * yaw_acceleration;
  q(idx::ROT_Z, idx::VYAW) += 0.5 * dt3 * yaw_acceleration;
  q(idx::VYAW, idx::ROT_Z) += 0.5 * dt3 * yaw_acceleration;
  q(idx::ROT_Z, idx::ROT_Z) += 0.25 * dt4 * yaw_acceleration;

  // ③ roll/pitch：随机游走。第三维给 0，yaw 的噪声已由 ② 负责。
  constexpr std::array<int, 3> rotation_index{idx::ROT_X, idx::ROT_Y, idx::ROT_Z};
  const Eigen::Vector3d roll_pitch_diagonal(config.roll_pitch, config.roll_pitch, 0.0);
  for (int i = 0; i < 3; ++i) {
    q(rotation_index[i], rotation_index[i]) += dt * roll_pitch_diagonal[i];
  }

  // ④ 半径与高度：随机游走。状态存的是 ln r，配置里的 radius 描述的是**物理
  //    半径**每秒能漂多少，所以要按一阶传播 σ_ℓ ≈ σ_r / r 换算，即除以 r²。
  //    副作用是大半径目标的对数噪声更小，符合直觉。
  const double r1 = std::exp(x[idx::LOG_R1]);
  q(idx::LOG_R1, idx::LOG_R1) = config.radius / (r1 * r1);

  if (outpost) {
    q(idx::OUTPOST_DZ1, idx::OUTPOST_DZ1) = config.outpost_height;
    q(idx::OUTPOST_DZ2, idx::OUTPOST_DZ2) = config.outpost_height;
  } else {
    const double r2 = std::exp(x[idx::LOG_R2]);
    q(idx::LOG_R2, idx::LOG_R2) = config.radius / (r2 * r2);
    q(idx::HEIGHT, idx::HEIGHT) = config.height;
  }

  return q;
}

}  // namespace L3Estimation::VehicleModel
