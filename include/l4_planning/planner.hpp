#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/aim_phase.hpp"
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
// 三处刻意偏离参考实现：
//
//  1. **不动点迭代对每块板各跑一次，收敛后再选板**（talos 的做法）。把选板
//     放进迭代里（sp / Climber 的做法）会让飞行时间在两块板之间来回跳：飞行
//     时间变了 → 外推的 yaw 变了 → 选中的板变了 → 距离变了 → 飞行时间又变。
//     在窗口边界附近这个循环根本不收敛，而那恰恰是最需要出解的时刻。逐板求
//     解则每次迭代的目标固定，必然收敛。
//
//  2. **plan_time 由调用方传入**，而不是内部取 steady_clock::now()。sp 的
//     Aimer::aim 在 to_now 分支里直接读当前时钟，同一段回放跑两次结果不同；
//     newvision 有离线回放 harness，必须保持 plan() 是纯函数。
//
//  3. **延迟仍按 Delay 的五段拆分记录**，不塌缩成一个标量。
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
    const std::optional<L3Estimation::TargetState>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time);

  // 目标丢失或切换车辆时调用，清掉选板锁定和档位状态。
  void reset() noexcept override;

  [[nodiscard]] PlanType type() const noexcept override { return PlanType::Setpoint; }

  [[nodiscard]] int lockedArmorId() const noexcept { return locked_id_; }
  [[nodiscard]] AimPhase aimPhase() const noexcept { return phase_.phase(); }
  [[nodiscard]] const PlanConfig& config() const noexcept { return config_; }

private:
  // 单块装甲板收敛后的候选。time 是从曝光时刻起算的总提前量（含击发前延迟
  // 和飞行时间），xyza 是该时刻这块板的位姿。
  struct Candidate {
    bool valid{false};
    int armor_id{-1};
    double total_time{0.0};
    double fly_time{0.0};
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};
    double delta_angle{0.0};
  };

  // 固定瞄第 armor_id 块板，对飞行时间做不动点迭代直到稳定。
  [[nodiscard]] Candidate refineArmor(
    const L3Estimation::TargetState& target, int armor_id, double before_fire,
    double bullet_speed) const;

  // 在收敛后的候选里选板：锁定优先，其次前置窗口内夹角最小，窗口全空时退化
  // 成全局夹角最小并置 degraded。降级也要给角度——云台停下来比指偏更糟。
  [[nodiscard]] int selectArmor(
    const std::vector<Candidate>& candidates, bool geometry_observed, bool& degraded);

  // WholeCarCenter 档的瞄准点：从枪口指向旋转中心的射线上，退回一个半径。
  [[nodiscard]] static Eigen::Vector3d projectCenterAim(
    const L3Estimation::TargetState& predicted, double radius, double armor_z);

  // 子弹飞到时这块板还正不正对枪口。coming/leaving 窗口，前哨站用宽窗口。
  [[nodiscard]] bool inFireWindow(
    const L3Estimation::TargetState& target, double delta_angle) const;

  PlanConfig config_;
  Predictor predictor_;
  BallisticSolver ballistic_;
  AimPhaseTracker phase_;
  int locked_id_{-1};
};

}  // namespace L4Planning
