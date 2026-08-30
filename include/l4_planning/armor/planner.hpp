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

class IPlanner {
public:
  virtual ~IPlanner() = default;

  [[nodiscard]] virtual Plan plan(const PlanInput& input) = 0;
  virtual void reset() noexcept = 0;
};

// 定点规划器：预测命中时刻、选择实体装甲板并解算 yaw/pitch。
class Planner final : public IPlanner {
public:
  explicit Planner(ArmorPlanConfig config = {});

  [[nodiscard]] Plan plan(const PlanInput& input) override;
  [[nodiscard]] Plan plan(
    const std::optional<L3Estimation::TrackedTarget>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time,
    bool to_now = true);

  void reset() noexcept override;
  int lockedArmorId() const noexcept { return locked_id_; }

private:
  struct AimPoint {
    bool valid{false};
    int armor_id{-1};  // armor_xyza_list() 中的物理板编号
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};  // [x, y, z, normal_yaw]
  };

  AimPoint chooseAimPoint(
    const L3Estimation::TrackedTarget& target);

  ArmorPlanConfig config_;
  int locked_id_{-1};
};

}  // namespace L4Planning
