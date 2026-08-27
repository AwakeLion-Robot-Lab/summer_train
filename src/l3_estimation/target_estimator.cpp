#include "l3_estimation/target_estimator.hpp"

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace {

// 车辆旋转半径的物理范围，仅用于发散判定。状态本身不做投影：sp_vision 任由
// 半径被单次观测拽出物理范围，再由 diverged() 把整个目标丢掉。
constexpr double kMinRadius = 0.05;
constexpr double kMaxRadius = 0.5;

// 计算笛卡尔坐标 [x, y, z] 到 [方位角, 俯仰角, 距离] 的 Jacobian。
Eigen::Matrix3d xyzToYpdJacobian(const Eigen::Vector3d &xyz) {
  const double x = xyz.x();
  const double y = xyz.y();
  const double z = xyz.z();
  const double xy_squared = x * x + y * y;
  const double distance_squared = xy_squared + z * z;

  // 球坐标在原点或 z 轴上不可导。正常 PnP 观测不会进入此分支，
  // 这里返回零矩阵以避免无效观测污染 EKF。
  if (xy_squared <= 1e-12 || distance_squared <= 1e-12) {
    return Eigen::Matrix3d::Zero();
  }

  const double xy = std::sqrt(xy_squared);
  const double distance = std::sqrt(distance_squared);

  Eigen::Matrix3d jacobian;
  jacobian << -y / xy_squared, x / xy_squared, 0.0,
      -x * z / (distance_squared * xy), -y * z / (distance_squared * xy),
      xy / distance_squared, x / distance, y / distance, z / distance;
  return jacobian;
}

} // namespace

namespace L3Estimation {

TrackedTarget::TrackedTarget(const Armor &armor,
                             std::chrono::steady_clock::time_point t,
                             double radius, int armor_num,
                             Eigen::VectorXd P0_dig)
    : name(armor.name), armor_type(armor.type), armor_num_(armor_num), t_(t) {
  if (armor_num_ < 1) {
    throw std::invalid_argument("armor_num must be positive");
  }
  if (P0_dig.size() != 11) {
    throw std::invalid_argument(
        "TrackedTarget requires an 11-element P0 diagonal");
  }

  const Eigen::Vector3d &xyz = armor.xyz_in_world;
  const double armor_yaw = armor.ypr_in_world[0];

  // 由当前装甲板位置反推旋转中心。
  const double center_x = xyz.x() + radius * std::cos(armor_yaw);
  const double center_y = xyz.y() + radius * std::sin(armor_yaw);
  const double center_z = xyz.z();

  // 内部状态：[xc, vx, yc, vy, z, vz, yaw, v_yaw, r1, r2-r1, z2-z1]。
  Eigen::VectorXd x0(11);
  x0 << center_x, 0.0, center_y, 0.0, center_z, 0.0, armor_yaw, 0.0, radius,
      0.0, 0.0;
  const Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // yaw 是周期量，每次注入滤波修正后都归一化到统一范围。半径**不做**投影：
  // 半径的 P0 是 1.0 m²（σ 达 1 米，而物理范围只有 5~50 厘米），先验极松，
  // 单次观测就能把 r 拽成负数。sp_vision 接受这一点，靠 diverged() 把越界的
  // 目标整个丢掉，而不是把状态夹回范围内继续跟。
  auto x_add = [](const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    Eigen::VectorXd result = a + b;
    result[6] = L6Telemetry::limit_rad(result[6]);
    return result;
  };

  ekf_ = ExtendedKalmanFilter(x0, P0, std::move(x_add));
}

TrackedTarget::TrackedTarget(double x, double vyaw, double radius,
                             double height, double yaw) {
  // 该构造入口直接给定部分运动状态，其余分量和初始协方差置零。
  Eigen::VectorXd x0(11);
  x0 << x, 0.0, 0.0, 0.0, 0.0, 0.0, L6Telemetry::limit_rad(yaw), vyaw, radius,
      0.0, height;
  const Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(11, 11);

  // 与上面的构造入口保持同一套流形运算，避免两条初始化路径行为分叉。
  auto x_add = [](const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
    Eigen::VectorXd result = a + b;
    result[6] = L6Telemetry::limit_rad(result[6]);
    return result;
  };

  ekf_ = ExtendedKalmanFilter(x0, P0, std::move(x_add));
}

void TrackedTarget::predict(std::chrono::steady_clock::time_point t) {
  // 绝对时间入口负责维护滤波器最后一次预测时刻。
  const double dt = L6Telemetry::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void TrackedTarget::predict(double dt) {
  // 位置和 yaw 采用恒速度模型，半径差与高度差在预测阶段保持不变。
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  double v1 = 100.0;
  double v2 = 400.0;
  // 前哨站运动模式更稳定，因此使用更小的平移和角速度过程噪声。
  if (name == ArmorName::Outpost) {
    v1 = 10.0;
    v2 = 0.1;
  }

  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;
  const double a = dt4 / 4.0;
  const double b = dt3 / 2.0;
  const double c = dt2;

  Eigen::Matrix2d constant_velocity_noise;
  constant_velocity_noise << a, b, b, c;

  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(11, 11);
  Q.block<2, 2>(0, 0) = v1 * constant_velocity_noise;  // x, vx
  Q.block<2, 2>(2, 2) = v1 * constant_velocity_noise;  // y, vy
  Q.block<2, 2>(4, 4) = v1 * constant_velocity_noise;  // z, vz
  Q.block<2, 2>(6, 6) = v2 * constant_velocity_noise;  // yaw, v_yaw

  auto transition = [&F](const Eigen::VectorXd &x) {
    Eigen::VectorXd prior = F * x;
    prior[6] = L6Telemetry::limit_rad(prior[6]);
    return prior;
  };

  // 收敛后将前哨站角速度吸附到已知转速量级，保留原有旋转方向。
  if (converged() && name == ArmorName::Outpost && std::abs(ekf_.x[7]) > 2.0) {
    ekf_.x[7] = ekf_.x[7] > 0.0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, transition);
}

void TrackedTarget::update(const Armor &armor) {
  // 先展开整车模型中的全部物理装甲板，再进行离散编号关联。
  int id = -1;
  double minimum_angle_error = std::numeric_limits<double>::infinity();
  const std::vector<Eigen::Vector4d> xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> candidates;
  candidates.reserve(xyza_list.size());
  for (int index = 0; index < armor_num_; ++index) {
    candidates.emplace_back(xyza_list[index], index);
  }

  // 优先评估离相机最近的装甲面，降低背面候选造成的错误关联。
  std::sort(candidates.begin(), candidates.end(),
            [](const std::pair<Eigen::Vector4d, int> &a,
               const std::pair<Eigen::Vector4d, int> &b) {
              return L6Telemetry::xyz2ypd(a.first.head<3>()).z() <
                     L6Telemetry::xyz2ypd(b.first.head<3>()).z();
            });

  // 最多取距离最近的 3 个候选装甲面；前哨站和基地本身就只有 3 个面。
  const int candidate_count = std::min(3, armor_num_);
  for (int index = 0; index < candidate_count; ++index) {
    const auto &xyza = candidates[index].first;
    const Eigen::Vector3d ypd = L6Telemetry::xyz2ypd(xyza.head<3>());
    // 关联代价同时考虑装甲板自身 yaw 和观测射线方位角。求和次序与 sp_vision
    // 一致，避免浮点结合律在两块板代价极接近时选出不同的板。
    const double angle_error =
        std::abs(L6Telemetry::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
        std::abs(L6Telemetry::limit_rad(armor.ypd_in_world.x() - ypd.x()));

    if (angle_error < minimum_angle_error) {
      id = candidates[index].second;
      minimum_angle_error = angle_error;
    }
  }

  if (id < 0) {
    L6Telemetry::logWarn(
        "TrackedTarget update skipped: no armor face candidate");
    return;
  }

  // 粘滞标志：只有真正观测到过 0 号以外的板，整车 yaw、r2-r1 和 z2-z1 才被
  // 数据约束过。一旦为真不再复位。
  if (id != 0) jumped = true;
  is_switch_ = id != last_id;
  if (is_switch_)
    ++switch_count_;

  last_id = id;
  ++update_count_;
  update_ypda(armor, id);
}

void TrackedTarget::update_ypda(const Armor &armor, int id) {
  // 四维观测为 [方位角, 俯仰角, 距离, 装甲板 yaw]。普通 EKF 只在先验点线性化
  // 一次，所以这里直接算好矩阵；换成迭代滤波器时需要改传 Jacobian 函数。
  const Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  const double center_yaw =
      std::atan2(armor.xyz_in_world.y(), armor.xyz_in_world.x());
  const double delta_angle =
      L6Telemetry::limit_rad(armor.ypr_in_world[0] - center_yaw);

  // 观测噪声取 sp_vision 的数值：方位角/俯仰角 4e-3（sigma 3.62 度），距离
  // 基底 1.0 m²（正视时 sigma 就有 1 米），两者都远大于实测噪声。
  //
  // 距离方差随 delta_angle 增长，编码的是"斜视时单板 PnP 深度精度迅速变差"；
  // 板 yaw 方差随距离增长。这两条随动关系与本项目此前的实现一致，差别只在基底。
  constexpr double kAngleVariance = 4e-3;
  constexpr double kDistanceVarianceFloor = 1.0;

  Eigen::VectorXd R_diagonal(4);
  const double armor_yaw_variance =
    std::log1p(std::abs(armor.ypd_in_world.z())) / 200.0 + 9e-2;
  const double distance_variance =
    kDistanceVarianceFloor + std::log1p(std::abs(delta_angle));
  R_diagonal << kAngleVariance, kAngleVariance, distance_variance, armor_yaw_variance;
  const Eigen::MatrixXd R = R_diagonal.asDiagonal();

  // 将十一维整车状态映射到指定物理装甲板的四维观测空间。
  auto observation = [this, id](const Eigen::VectorXd &x) {
    const Eigen::Vector3d xyz = h_armor_xyz(x, id);
    const Eigen::Vector3d ypd = L6Telemetry::xyz2ypd(xyz);
    const double angle =
        L6Telemetry::limit_rad(x[6] + id * 2.0 * std::numbers::pi / armor_num_);
    return Eigen::Vector4d{ypd.x(), ypd.y(), ypd.z(), angle};
  };

  // 三个角度残差都必须走最短圆周差，距离分量保持普通减法。
  auto subtract_observation = [](const Eigen::VectorXd &a,
                                 const Eigen::VectorXd &b) {
    Eigen::VectorXd result = a - b;
    result[0] = L6Telemetry::limit_rad(result[0]);
    result[1] = L6Telemetry::limit_rad(result[1]);
    result[3] = L6Telemetry::limit_rad(result[3]);
    return result;
  };

  Eigen::VectorXd z(4);
  z << armor.ypd_in_world.x(), armor.ypd_in_world.y(), armor.ypd_in_world.z(),
      armor.ypr_in_world[0];
  ekf_.update(z, H, R, observation, subtract_observation);
}

Eigen::VectorXd TrackedTarget::ekf_x() const { return ekf_.x; }

const ExtendedKalmanFilter &TrackedTarget::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> TrackedTarget::armor_xyza_list() const {
  std::vector<Eigen::Vector4d> armors;
  // 物理装甲板绕中心等角分布；四板车奇数板使用第二组半径和高度。
  armors.reserve(armor_num_);
  for (int id = 0; id < armor_num_; ++id) {
    const double angle = L6Telemetry::limit_rad(
        ekf_.x[6] + id * 2.0 * std::numbers::pi / armor_num_);
    const Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, id);
    armors.emplace_back(xyz.x(), xyz.y(), xyz.z(), angle);
  }
  return armors;
}

bool TrackedTarget::diverged() const {
  // 瞬时判据：r 或 r2 一旦离开物理范围就把整个目标作废。状态不做投影，所以
  // 这是唯一的约束手段——半径越界即表示观测与整车模型无法调和。
  const double r1 = ekf_.x[8];
  const double r2 = ekf_.x[8] + ekf_.x[9];
  const bool r1_ok = r1 > kMinRadius && r1 < kMaxRadius;
  const bool r2_ok = r2 > kMinRadius && r2 < kMaxRadius;
  if (r1_ok && r2_ok) {
    return false;
  }

  L6Telemetry::logDebug("TrackedTarget diverged: r1, r2", r1, r2);
  return true;
}

bool TrackedTarget::converged() {
  // 前哨站需要更多更新帧，其余车辆使用较短确认窗口。
  const int required_updates = name == ArmorName::Outpost ? 10 : 3;
  if (update_count_ > required_updates && !diverged()) {
    is_converged_ = true;
  }
  return is_converged_;
}

Eigen::Vector3d TrackedTarget::h_armor_xyz(const Eigen::VectorXd &x,
                                           int id) const {
  // 编号 id 决定装甲板绕中心的离散相位。
  const double angle =
      L6Telemetry::limit_rad(x[6] + id * 2.0 * std::numbers::pi / armor_num_);
  const bool use_alternate_radius = armor_num_ == 4 && (id == 1 || id == 3);

  const double radius = use_alternate_radius ? x[8] + x[9] : x[8];
  const double armor_x = x[0] - radius * std::cos(angle);
  const double armor_y = x[2] - radius * std::sin(angle);
  const double armor_z = use_alternate_radius ? x[4] + x[10] : x[4];
  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd TrackedTarget::h_jacobian(const Eigen::VectorXd &x,
                                          int id) const {
  const double angle =
      L6Telemetry::limit_rad(x[6] + id * 2.0 * std::numbers::pi / armor_num_);
  const bool use_alternate_radius = armor_num_ == 4 && (id == 1 || id == 3);

  const double radius = use_alternate_radius ? x[8] + x[9] : x[8];
  const double dx_da = radius * std::sin(angle);
  const double dy_da = -radius * std::cos(angle);
  const double dx_dr = -std::cos(angle);
  const double dy_dr = -std::sin(angle);
  const double dx_dl = use_alternate_radius ? -std::cos(angle) : 0.0;
  const double dy_dl = use_alternate_radius ? -std::sin(angle) : 0.0;
  const double dz_dh = use_alternate_radius ? 1.0 : 0.0;

  // 先求整车状态到 [armor_x, armor_y, armor_z, armor_yaw] 的 Jacobian。
  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  // 再与 xyz -> ypd 的 Jacobian 链乘，得到最终四维观测 Jacobian。
  const Eigen::Vector3d armor_xyz = h_armor_xyz(x, id);
  const Eigen::Matrix3d H_armor_ypd = xyzToYpdJacobian(armor_xyz);
  Eigen::Matrix4d H_armor_ypda = Eigen::Matrix4d::Zero();
  H_armor_ypda.topLeftCorner<3, 3>() = H_armor_ypd;
  H_armor_ypda(3, 3) = 1.0;
  return H_armor_ypda * H_armor_xyza;
}

} // namespace L3Estimation

