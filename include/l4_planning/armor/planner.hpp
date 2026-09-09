#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/armor/target_estimator.hpp"
#include "l4_planning/armor/types.hpp"

#include <Eigen/Core>

#include <optional>

namespace L4Planning {

struct PlanInput {
  std::optional<L3Estimation::TrackedTarget> target;
  L1Sensor::RobotState robot_state;
  TimePoint plan_time{};  // 本次规划开始的 steady_clock 时间
  bool to_now{true};      // 是否补偿 target.t() 到 plan_time 的已发生延迟
  // 上一帧实测的"规划结束 -> 串口发出"耗时，单位秒。本帧的值要等规划做完
  // 才知道，所以只能用上一帧的量代入；帧间这一段基本恒定。
  double plan_to_send{0.0};
};

// 定点规划器：预测命中时刻、选择实体装甲板并解算 yaw/pitch。
class Planner final {
public:
  explicit Planner(ArmorPlanConfig config = {});

  [[nodiscard]] Plan plan(const PlanInput& input);
  [[nodiscard]] Plan plan(
    const std::optional<L3Estimation::TrackedTarget>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time,
    bool to_now = true);

  void reset() noexcept;
  int lockedArmorId() const noexcept { return locked_id_; }
  bool blending() const noexcept { return smoother_.blending(); }

private:
  struct AimPoint {
    bool valid{false};
    int armor_id{-1};  // armor_xyza_list() 中的物理板编号
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};  // [x, y, z, normal_yaw]
  };

  // 纯函数：迟滞锁只经 lock 进出，不写成员。弹道迭代要在同一帧里反复调用
  // 它去试探不同的假想构型，一旦它自己改成员，跨帧的迟滞状态就会被中间轮次
  // 覆盖，最终锁值取决于迭代恰好在第几轮收敛。
  AimPoint chooseAimPoint(
    const L3Estimation::TrackedTarget& target, int& lock) const;

  // 射击轨迹在"本帧之后 offset 秒"的取值：把发射时刻的状态再推
  // fly_time + offset，展开出**指定**物理板（不是重新选板——过渡段的终点
  // 必须钉在切板后那一块上），解一次弹道得到 yaw/pitch。
  //
  // 飞行时间沿用本帧收敛值、不再逐样本迭代：前视窗口只有两百毫秒，这期间
  // 飞行时间的变化远小于整车 yaw 转过的角度，而迭代会把采样成本乘上
  // max_iterations。
  //
  // 求导走中心差分，且差分必须先归一化：aim yaw 出自 atan2，天然落在
  // (-pi, pi]，直接相减会在 ±pi 处得到一个 2pi/dt 的假尖峰，那正好会被
  // 当成"需要无穷大加速度"。失败时返回全 NaN，由 fitBlend 的有限性检查挡下。
  AimState sampleTrajectory(
    const L3Estimation::TrackedTarget& target_at_fire,
    double fly_time,
    double bullet_speed,
    double offset,
    int armor_id) const;

  // 前视扫描，找选板结果首次改变的时刻。空窗（本帧没有任何可击打板）不算
  // 切板：前哨站两块板之间就有这么一段，过渡段应当盖住它，而不是在那里
  // 重新起跑。
  std::optional<double> nextSwitchTime(
    const L3Estimation::TrackedTarget& target_at_fire,
    double fly_time,
    int current_id,
    int& next_id) const;

  // 过渡段进行中、但本帧选不出可击打板时的降级计划：继续把过渡段发下去，
  // 而不是让 L5 走 safeHold 把云台角冻住。没有实体板可判，所以只能 TrackOnly。
  std::optional<Plan> blendOnlyPlan(TimePoint now, const Delay& delay);

  ArmorPlanConfig config_;
  // 弹道求解器在构造时按 ballistic.drag_coefficient 选定模型：0 走真空闭式
  // 解，非 0 走等效距离的二次阻力闭式解。每帧解算不再重建模型。
  BallisticSolver ballistic_;
  int locked_id_{-1};

  AimSmoother smoother_;
  // 最近一次解出的射击轨迹角。空窗帧里过渡段还要继续求值，但那一帧没有
  // 新的射击轨迹可算，只能沿用上一次的原值去填 shoot_yaw/shoot_pitch。
  double last_shoot_yaw_{0.0};
  double last_shoot_pitch_{0.0};
  bool has_last_shoot_{false};
};

}  // namespace L4Planning
