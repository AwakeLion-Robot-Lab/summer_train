#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"

#include <optional>

namespace L4Planning {

// 规划器输入。用结构体而不是参数列表，是为了后续加字段时不必改动已有
// 实现的签名——MPC 和五次多项式需要的边界条件比定点规划器多。
struct PlanInput {
  // L3 在曝光时刻的整车快照，空表示当前无目标。
  std::optional<L3Estimation::TargetState> target;
  L1Sensor::RobotState robot_state;

  // 本次规划发生的时刻。必须由调用方传入而不是内部取 now()：离线回放要
  // 求 plan() 是纯函数，同一段数据跑两次必须得到同一结果。
  TimePoint plan_time{};

  // 云台当前角速度，作为轨迹起点的边界条件。定点规划器忽略这两个字段；
  // 五次多项式和 MPC 需要它们才能保证轨迹在起点速度连续。下位机尚未回传
  // 时保持 nullopt，实现方应当据此退化而不是假定为 0。
  std::optional<double> gimbal_yaw_vel;
  std::optional<double> gimbal_pitch_vel;
};

// 规划器接口。三种实现共用同一份输入输出契约：
//
//   Setpoint       当前实现。只解命中点，速度和加速度保持 0。
//   QuinticSwitch  只在小陀螺切板造成轨迹断点处插入五次多项式过渡段，
//                  跟随段仍贴合原射击轨迹。用两端的位置、速度、加速度
//                  六个边界条件定系数，逐步增大过渡时间直到最大加速度
//                  低于云台极限。
//   TinyMpc        全程用 MPC 约束云台角加速度。
//
// 三者的差别只在 Plan 的 yaw_vel / pitch_vel / yaw_acc / pitch_acc 是否
// 被填充，以及 type 字段。L5 不需要知道用的是哪一种。
class IPlanner {
public:
  virtual ~IPlanner() = default;

  IPlanner() = default;
  IPlanner(const IPlanner&) = delete;
  IPlanner& operator=(const IPlanner&) = delete;
  IPlanner(IPlanner&&) = delete;
  IPlanner& operator=(IPlanner&&) = delete;

  [[nodiscard]] virtual Plan plan(const PlanInput& input) = 0;

  // 目标丢失或切换车辆时调用。轨迹类规划器还需要在这里丢弃过渡段状态。
  virtual void reset() noexcept = 0;

  [[nodiscard]] virtual PlanType type() const noexcept = 0;
};

}  // namespace L4Planning
