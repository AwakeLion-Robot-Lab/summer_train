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

  ArmorPlanConfig config_;
  int locked_id_{-1};
};

}  // namespace L4Planning
