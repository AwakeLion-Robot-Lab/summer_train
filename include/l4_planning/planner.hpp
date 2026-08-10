#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/aim_phase.hpp"
#include "l4_planning/planner_interface.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Core>

#include <optional>

namespace L4Planning {

// 逐项复刻 sp_vision 的 Aimer：预发射预测 → 选板 → 初始弹道 → 在每轮共同
// 命中时刻重新预测、重新选板并更新锁。Plan 只是承接结果的 newvision 外壳，
// 不改变 SP 的选板与迭代顺序。
//
// 定点规划器：只解出命中时刻的瞄准角，速度和加速度保持 0。QuinticSwitch 和
// TinyMpc 实现同一个 IPlanner 接口，届时填充 Plan 里的 yaw_vel / pitch_vel /
// yaw_acc / pitch_acc，L5 无需改动。
class Planner final : public IPlanner {
public:
  explicit Planner(PlanConfig config = {});

  [[nodiscard]] Plan plan(const PlanInput& input) override;

  // 便捷重载，等价于填一个只有三个字段的 PlanInput。
  [[nodiscard]] Plan plan(
    const std::optional<L3Estimation::TrackedTarget>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time,
    bool to_now = true);

  // SP Aimer 没有 reset；锁只能由 chooseAimPoint 的普通车单候选分支清除。
  void reset() noexcept override;

  [[nodiscard]] PlanType type() const noexcept override { return PlanType::Setpoint; }

  [[nodiscard]] int lockedArmorId() const noexcept { return locked_id_; }
  [[nodiscard]] AimPhase aimPhase() const noexcept { return AimPhase::SingleArmor; }
  [[nodiscard]] const PlanConfig& config() const noexcept { return config_; }

private:
  struct AimPoint {
    bool valid{false};
    int armor_id{-1};
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};
    double delta_angle{0.0};
  };

  [[nodiscard]] AimPoint chooseAimPoint(
    const L3Estimation::TrackedTarget& target);

  PlanConfig config_;
  int locked_id_{-1};
};

}  // namespace L4Planning
