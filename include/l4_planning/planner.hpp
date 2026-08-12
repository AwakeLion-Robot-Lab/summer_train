#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/tracked_target.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Core>

#include <optional>

namespace L4Planning {

// 规划器输入。用结构体而不是参数列表，后续加边界条件（云台角速度之类）时
// 不必改动调用点。
struct PlanInput {
  // L3 在曝光时刻的整车目标副本，空表示当前无目标。规划器可以在它上面自由
  // 外推——那是自己的副本，不会影响 Tracker 持有的滤波器。
  std::optional<L3Estimation::TrackedTarget> target;
  L1Sensor::RobotState robot_state;

  // 本次规划发生的时刻。由调用方传入而不是内部取 now()：离线回放要求
  // plan() 是纯函数，同一段数据跑两次必须得到同一结果。
  TimePoint plan_time{};

  // 实机为 true，用曝光到 plan_time 的实测延迟；离线回放为 false，用固定的
  // 检测耗时估计，保证结果与运行速度无关。
  bool to_now{true};
};

// 定点规划器：把整车 EKF 外推到弹丸命中的时刻，挑一块装甲板，解出云台该指向
// 的 yaw / pitch。
//
// 一次 plan() 的顺序是
//   1. 按目标转速选延迟档 -> 外推到击发时刻
//   2. 选板（chooseAimPoint）
//   3. 解弹道拿到飞行时间 -> 外推到命中时刻 -> 重新选板重新解，直到飞行时间收敛
//   4. 由命中时刻的瞄准点算出 yaw / pitch
class Planner {
public:
  explicit Planner(PlanConfig config = {});

  [[nodiscard]] Plan plan(const PlanInput& input);

  // 便捷重载，等价于填一个只有三个字段的 PlanInput。
  [[nodiscard]] Plan plan(
    const std::optional<L3Estimation::TrackedTarget>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time,
    bool to_now = true);

  [[nodiscard]] int lockedArmorId() const noexcept { return locked_id_; }
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
  // 锁定的装甲板编号，防止在两块夹角相近的板之间来回横跳。
  int locked_id_{-1};
};

}  // namespace L4Planning
