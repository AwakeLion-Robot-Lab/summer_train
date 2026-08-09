#pragma once

#include "l3_estimation/ekf_.hpp"
#include "l3_estimation/types.hpp"

#include <chrono>
#include <vector>

namespace L3Estimation {

// L3 内部维护的整车 EKF。内部十一维状态为
// [xc, vx, yc, vy, z, vz, yaw, v_yaw, r1, r2-r1, z2-z1]。
// 跨层输出使用 TargetState 快照，外部不能直接修改滤波器状态。
class TrackedTarget
{
public:
  // 目标类别、板型以及最近一次装甲板关联结果。
  ArmorName name{ArmorName::Unknown};
  ArmorType armor_type{ArmorType::Small};
  // jumped 表示本次观测关联到的不是编号 0 的装甲板。
  bool jumped{false};
  // multi_armor_observed 是 jumped 的粘滞版本：只要关联到过 0 号以外的板就
  // 一直为 true。两者语义不同——jumped 回答"这一帧看的是哪块板"，
  // multi_armor_observed 回答"整车几何到底可不可观测"。L4 需要的是后者。
  bool multi_armor_observed{false};
  int last_id{0};

  TrackedTarget() = default;
  // 使用首个装甲板观测反推旋转中心并初始化十一维状态。
  // sp_compat 逐项把行为切回 sp_vision 的实现，仅用于差分定位，见 SpCompatConfig。
  TrackedTarget(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, SpCompatConfig sp_compat = {});
  // 构造指定旋转状态的目标，主要用于无观测的确定性初始化。
  TrackedTarget(double x, double vyaw, double radius, double h);

  // 按绝对时间或显式时间间隔执行恒速度预测。
  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  // 关联观测对应的物理装甲板，并执行一次 EKF 更新。
  void update(const Armor & armor);

  // 提供只读滤波结果和由整车模型展开的 [x, y, z, yaw] 装甲板列表。
  Eigen::VectorXd ekf_x() const;
  const ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  // 生成交给 L4 的只含数据的快照。
  [[nodiscard]] TargetState toTargetState(
    TrackState track_state, bool updated = true) const;

  // 任一候选半径离开物理范围时认为滤波器发散。
  bool diverged() const;

  // 有效更新达到门限且状态未发散后，目标保持收敛标志。
  bool converged();

  bool isinit{false};

  [[nodiscard]] bool checkinit() const noexcept;

private:
  // 车辆物理装甲板数量以及关联、收敛统计。
  int armor_num_{4};
  int switch_count_{0};
  int update_count_{0};
  // 半径被投影顶在物理边界上的连续更新次数，超过门限视为发散。
  int radius_pinned_count_{0};
  // sp_vision 行为复刻开关，构造后不再变化。
  SpCompatConfig sp_compat_{};

  bool is_switch_{false};
  bool is_converged_{false};

  // 主干路使用单次线性化的普通 EKF。迭代实现见 ieskf.hpp，接回的方法写在
  // docs/iterated_ekf.md，改动只涉及本成员的类型和 update_ypda 的传参。
  ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};

  // 使用 [方位角, 俯仰角, 距离, 装甲板 yaw] 观测更新指定物理装甲板。
  void update_ypda(const Armor & armor, int id);

  // 从整车状态计算指定装甲板的位置及其观测 Jacobian。
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};
}  // namespace L3Estimation
