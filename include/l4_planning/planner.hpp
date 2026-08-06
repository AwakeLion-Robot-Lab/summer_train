#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/planner_interface.hpp"
#include "l4_planning/predictor.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Core>

#include <optional>
#include <vector>

namespace L4Planning {

// 结构对标 sp_vision 的 Aimer：预测 → 选板 → 弹道，并对飞行时间做不动点
// 迭代。选板作为私有方法内聚在这里，不单独立类。
//
// 与 sp 的两处刻意差异：
//  1. plan_time 由调用方传入，而不是内部取 steady_clock::now()。sp 的
//     Aimer::aim 在 to_now 分支里直接读当前时钟，同一段回放跑两次结果不
//     同；newvision 有离线回放 harness，必须保持 plan() 是纯函数。
//  2. 延迟仍按 Delay 的五段拆分记录，不塌缩成一个标量。
// 定点规划器：只解出命中时刻的瞄准角，速度和加速度保持 0。
// QuinticSwitch 和 TinyMpc 实现同一个 IPlanner 接口，届时填充 Plan 里的
// yaw_vel / pitch_vel / yaw_acc / pitch_acc，L5 无需改动。
class Planner final : public IPlanner {
public:
  explicit Planner(PlanConfig config = {});

  [[nodiscard]] Plan plan(const PlanInput& input) override;

  // 便捷重载，等价于填一个只有三个字段的 PlanInput。
  [[nodiscard]] Plan plan(
    const std::optional<L3Estimation::TargetState>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time);

  // 目标丢失或切换车辆时调用，清掉选板锁定状态。
  void reset() noexcept override;

  [[nodiscard]] PlanType type() const noexcept override { return PlanType::Setpoint; }

  [[nodiscard]] int lockedArmorId() const noexcept { return locked_id_; }
  [[nodiscard]] const PlanConfig& config() const noexcept { return config_; }

private:
  // 选中的瞄准点。delta_angle 是该板法线与观测方向的夹角，0 为正对枪口。
  struct AimPoint {
    bool valid{false};
    int armor_id{-1};
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};
    double delta_angle{0.0};
  };

  // 对应 sp 的 choose_aim_point。低速档挑正对的板并加迟滞，反陀螺档只打
  // 正在转入视野的一侧。有状态：locked_id_ 必须跨帧保持。
  [[nodiscard]] AimPoint chooseAimPoint(
    const L3Estimation::TargetState& target,
    const std::vector<Eigen::Vector4d>& armors);

  PlanConfig config_;
  Predictor predictor_;
  BallisticSolver ballistic_;
  int locked_id_{-1};
};

}  // namespace L4Planning
