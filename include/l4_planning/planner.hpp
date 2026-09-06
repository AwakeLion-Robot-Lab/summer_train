#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/latency_compensator.hpp"
#include "l4_planning/predictor.hpp"
#include "l4_planning/state.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Geometry>

#include <optional>
#include <vector>

namespace L4Planning {

struct PlannerContext {
  TimePoint planning_time{}; // 本周期开始规划的绝对时刻
  LatencyConfig latency;     // 系统延迟补偿配置
  // 世界系和枪口系轴向平行；仅使用该变换的平移部分。
  Eigen::Isometry3d T_barrel_world{Eigen::Isometry3d::Identity()};
  PlannerConfig config;                // 迭代、弹道和轨迹配置
  ArmorScoreWeights armor_score_weights; // 多装甲板评分权重
  // Q_facing 的平滑归一化区间，单位 degree。
  double facing_angle_good{5.0};  // 小于该角度时评分为 1
  double facing_angle_bad{25.0};  // 大于该角度时评分为 0
};

class Planner {
public:
  // 使用默认 PlannerConfig：真空弹道、最多 20 次固定点迭代。
  Planner() = default;
  // 构造时指定弹道模型和迭代参数。
  explicit Planner(PlannerConfig config);

  // 运行时替换配置；调用者应避免与 plan() 并发写读。
  void setConfig(PlannerConfig config);
  // 获取当前只读配置。
  [[nodiscard]] const PlannerConfig& config() const noexcept;
  // 清除跨帧目标和装甲板锁定状态。
  void resetTracking() noexcept;
  // 获取当前装甲板跟踪状态，供调试和遥测使用。
  [[nodiscard]] const ArmorTrackingState& trackingState() const noexcept;

  // 兼容入口：使用内部 PlannerConfig 和默认物理参数构造 Context。
  AimPlan plan(const std::optional<L3Estimation::TargetState>& target, const L1Sensor::RobotState& robot_state);
  // 主入口：使用本周期 Context 中的 PlannerConfig，对四块装甲板分别
  // 迭代并选择有效的最佳跟踪目标；射击窗口仅用于最终开火门控。
  AimPlan plan(
    const std::optional<L3Estimation::TargetState>& target,
    const L1Sensor::RobotState& robot_state,
    const PlannerContext& context);

private:
  [[nodiscard]] SelectionResult selectArmor(
    const SelectionRequest& request,
    TimePoint selection_time,
    const PlannerConfig& config);

  PlannerConfig config_; // 当前 Planner 使用的模型和收敛参数
  ArmorTrackingState tracking_state_;
  std::optional<L3Estimation::TargetState> last_target_;
  std::optional<TimePoint> last_observation_timestamp_;
  int target_lost_frames_{0};
};

}  
