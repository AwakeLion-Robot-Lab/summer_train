#pragma once

#include "l3_estimation/armor/eskf_target.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <optional>

namespace L4Planning {

struct PlanInput {
  std::optional<L3Estimation::EskfTarget> target;
  // 下位机回传的弹速，m/s。没发（0）或不可信时 Planner 换成 default_bullet_speed。
  double bullet_speed{0.0};
  TimePoint plan_time{};  // 本次规划开始的 steady_clock 时间
  bool to_now{true};      // 是否补偿 target.t() 到 plan_time 的已发生延迟
  // 上一帧实测的"规划结束 -> 串口发出"耗时，单位秒。本帧的值要等规划做完
  // 才知道，所以只能用上一帧的量代入；帧间这一段基本恒定。
  double plan_to_send{0.0};
  // 规划时刻的枪管姿态。只在进自瞄后的头一次选板用：两块板都能打时挑离枪口
  // 最近的那块。缺省时照常选更正对的板。
  std::optional<Eigen::Quaterniond> q_world_barrel;
};

// 定点规划器：预测命中时刻、选择实体装甲板并解算 yaw/pitch。
class Planner {
public:
  explicit Planner(PlanConfig config = {});

  [[nodiscard]] Plan plan(const PlanInput& input);

  // 标记一轮新的自瞄：下一次成功规划改按离枪口最近选板，见 choose。
  void reset() noexcept;

private:
  struct AimPoint {
    bool valid{false};
    int armor_id{-1};  // armor_xyza_list() 中的物理板编号
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()};  // [x, y, z, normal_yaw]
  };

  // muzzle 是枪口在世界系的单位指向，只在进自瞄后的头一次选板给。
  AimPoint choose(
    const L3Estimation::EskfTarget& target,
    const std::optional<Eigen::Vector3d>& muzzle);
  // 下位机弹速不可信时换成缺省值；只在切换时打一条日志。
  double bulletSpeed(double reported);

  PlanConfig config_;
  int locked_id_{-1};
  // 构造即算一轮新的自瞄，reset() 再置回；第一次成功规划后清掉。
  bool entering_{true};
  bool default_speed_{false};
};

}  // namespace L4Planning
