#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l3_estimation/armor/eskf_target.hpp"
#include "l4_planning/armor/types.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>

namespace L4Planning {

struct PlanInput {
  std::optional<L3Estimation::EskfTarget> target;
  L1Sensor::RobotState robot_state;
  TimePoint plan_time{};  // 本次规划开始的 steady_clock 时间
  bool to_now{true};      // 是否补偿 target.t() 到 plan_time 的已发生延迟
  // 上一帧实测的"规划结束 -> 串口发出"耗时，单位秒。本帧的值要等规划做完
  // 才知道，所以只能用上一帧的量代入；帧间这一段基本恒定。
  double plan_to_send{0.0};
  // 规划时刻的枪管姿态。只在进自瞄后的头一次选板用：两块板都能打时挑离枪口
  // 最近的那块。缺省时照常选更正对的板。
  std::optional<Eigen::Quaterniond> q_world_barrel;
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
    const std::optional<L3Estimation::EskfTarget>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time,
    bool to_now = true);
  // std::nullopt 的精确匹配，语义是 NoTarget。
  [[nodiscard]] Plan plan(
    std::nullopt_t,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time,
    bool to_now = true);

  // 标记一轮新的自瞄：下一次成功规划改按离枪口最近选板，见 chooseAimPoint。
  void reset() noexcept override;
  int lockedArmorId() const noexcept { return locked_id_; }

private:
  struct AimPoint {
    bool valid{false};
    int armor_id{-1};  // armor_xyza_list() 中的物理板编号
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};  // [x, y, z, normal_yaw]
  };

  template <typename Target>
  [[nodiscard]] Plan planTarget(
    const std::optional<Target>& target,
    const L1Sensor::RobotState& robot_state,
    TimePoint plan_time,
    bool to_now,
    double plan_to_send,
    const std::optional<Eigen::Quaterniond>& q_world_barrel);

  // muzzle 是枪口在世界系的单位指向，只在进自瞄后的头一次选板给。
  template <typename Target>
  AimPoint chooseAimPoint(
    const Target& target, const std::optional<Eigen::Vector3d>& muzzle);

  ArmorPlanConfig config_;
  int locked_id_{-1};
  // 构造即算一轮新的自瞄，reset() 再置回；第一次成功规划后清掉。
  bool entering_{true};
};

}  // namespace L4Planning
