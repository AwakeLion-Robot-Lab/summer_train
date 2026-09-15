#pragma once

#include "l3_estimation/ekf.hpp"
#include "l3_estimation/armor/types.hpp"

#include <chrono>
#include <vector>

namespace L3Estimation {

// 板间高度差。四板车只用 z2_z1（奇数板整体抬高）；三板车只用 dz1/dz2
// （1、2 号板各自相对 0 号板）。
struct HeightOffsets
{
  double z2_z1{0.0};
  double dz1{0.0};
  double dz2{0.0};
};

// 整车 EKF，前十一维对齐 sp_vision 的 auto_aim::Target，末两维是三板车的
// 板间高度差，完整顺序见 kStateSize 处。
//
// 本类型**就是** L3 交给 L4 的对象：Tracker::track() 返回它的副本，L4 在副本
// 上 predict(dt) 外推再用 armor_xyza_list() 展开。所以没有独立的跨层快照类型。
class TrackedTarget
{
public:
  // 状态维度。前十一维是 sp_vision 的整车模型，L4 按下标读它们，顺序不可改；
  // 末两维是三板车（2026 规则的前哨站、基地）1、2 号板相对 0 号板的高度差，
  // 四板车恒为 0。新增状态一律往后追加。
  static constexpr int kStateSize = 13;

  // 目标类别、板型以及最近一次装甲板关联结果。
  ArmorName name{ArmorName::Unknown};
  ArmorType armor_type{ArmorType::Small};
  // 是否关联到过编号 0 以外的装甲板。**粘滞**：一旦为真不再复位。
  // 为 false 时整车 yaw、第二组半径和高度差几乎不可观测——只见过一块板的话，
  // 其余板的位置完全由初值猜出来，所以 L4 据此只瞄当前观测到的那块板。
  bool jumped{false};
  int last_id{0};  // debug only

  // 没有默认构造：TrackedTarget 一经存在，状态就是完整的 kStateSize 维。
  // 这样下游不必到处验维度，也不会出现"半个目标"。
  // 需要可空语义时用 std::optional<TrackedTarget>。
  //
  // 使用首个装甲板观测反推旋转中心并初始化整个状态。
  TrackedTarget(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, TargetConfig config = {});
  // 构造指定构型的目标，用于无观测的确定性初始化（离线回放和单测）。
  // 板数由 name 推出，而不是再单独传一个——否则 name 和板数可以各说各话。
  // 旋转中心落在 (x, 0, 0)。yaw 在 sp_vision 的同名入口里固定为 0，这里放开
  // 成可选参数，否则测不到"整车转到某个角度"的构型。
  TrackedTarget(
    ArmorName name, double x, double vyaw, double radius, double yaw = 0.0,
    HeightOffsets heights = {}, TargetConfig config = {});

  // 按绝对时间或显式时间间隔执行恒速度预测。
  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  // 关联观测对应的物理装甲板，并执行一次 EKF 更新。
  void update(const Armor & armor);

  // 提供只读滤波结果和由整车模型展开的 [x, y, z, yaw] 装甲板列表。
  Eigen::VectorXd ekf_x() const;
  const ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // 车辆物理装甲板数量，以及滤波器最后一次推进到的时刻。L4 用后者算曝光到
  // 规划的可测量延迟段。
  int armor_num() const noexcept { return armor_num_; }
  std::chrono::steady_clock::time_point t() const noexcept { return t_; }

  // 任一候选半径离开物理范围时认为滤波器发散。
  bool diverged() const;

  // 有效更新达到门限且状态未发散后，目标保持收敛标志。
  bool converged();

private:
  // 过程噪声与观测噪声，由 Tracker 从 auto_aim.yaml 透传。
  TargetConfig config_{};
  // 车辆物理装甲板数量以及关联、收敛统计。
  int armor_num_{4};
  int switch_count_{0};
  int update_count_{0};

  bool is_switch_{false};
  bool is_converged_{false};

  ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};

  // 使用 [方位角, 俯仰角, 距离, 装甲板 yaw] 观测更新指定物理装甲板。
  void update_ypda(const Armor & armor, int id);

  // 该编号的板用哪一维高度偏移；-1 表示直接用中心高度 x[4]。
  int heightIndex(int id) const noexcept;
  // 从整车状态计算指定装甲板的位置及其观测 Jacobian。
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};
}  // namespace L3Estimation
