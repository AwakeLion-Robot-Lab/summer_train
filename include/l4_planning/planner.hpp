#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/latency_compensator.hpp"
#include "l4_planning/predictor.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Geometry>

#include <optional>
#include <vector>

namespace L4Planning {

// 装甲板质量评分的权重。四项权重之和为 1。
  struct ArmorScoreWeights {
    double facing_weight{0.30};
    double window_weight{0.30};
    double prediction_confidence_weight{0.25};
    double ballistic_weight{0.15};
  };

// 四项归一化质量分量，取值范围均为 [0, 1]
struct ArmorScoreComponents {
  double Q_facing{0.0};
  double Q_window{0.0};
  double Q_prediction_confidence{0.0};
  double Q_ballistic{0.0};
};

// flag
struct ArmorScoreHardConditions {
  bool identity_consistent{false};
  bool stable_tracking{false};
  bool prediction_valid{false};
  bool within_firing_window{false};
  bool ballistic_valid{false};
  bool iteration_converged{false};
};

// 一块候选装甲板的完整评分结果
//score（i） = flag(i) * Q(i)
struct ArmorScore {
  ArmorScoreComponents components;
  ArmorScoreHardConditions hard_conditions;
  bool flag{false};
  double quality{0.0};
  double score{0.0};
};

struct PlannerContext {
  TimePoint planning_time{}; // 本周期开始规划的绝对时刻
  LatencyConfig latency;     // 系统延迟补偿配置
  // 世界系和枪口系轴向平行；仅使用该变换的平移部分。
  Eigen::Isometry3d T_barrel_world{Eigen::Isometry3d::Identity()};
  PlannerConfig config;                // 迭代、弹道和轨迹配置
  ArmorScoreWeights armor_score_weights; // 多装甲板评分权重
};

struct ArmorCandidate {
  ArmorPose armor;                 // 迭代收敛后的装甲板世界系位姿
  BallisticSolution ballistic;    // 对最终装甲板位置的弹道解
  TimePoint impact_time{};        // 预测弹丸命中装甲板的绝对时刻
  double delta_angle{0.0};        // 装甲板法向与目标方位角之差，rad
  int iteration_count{0};         // 固定点迭代实际执行次数
  double fly_time_error{0.0};     // 相邻两次飞行时间之差，s
  double position_error{0.0};     // 相邻两次预测位置之差，m
  double angle_error{0.0};        // 相邻两次瞄准角的二维误差，rad
  bool converged{false};          // 时间误差和位置/角度误差是否收敛
  bool within_firing_window{false}; // 命中时刻是否仍在可射击窗口
  bool valid{false};              // 所有硬条件是否同时满足
  ArmorScore score;               // 用于多装甲板选择的最终评分
};

enum class SelectionReason {
  NoCandidate,
  PreferredArmor,
  EnteringArmor,
  MostFacing,
  BallisticFeasible
};

struct SelectionRequest {
  std::vector<ArmorCandidate> candidates;
  std::optional<int> preferred_armor_id;
};

struct SelectionResult {
  std::optional<ArmorCandidate> selected;
  bool switching{false};
  bool valid{false};
  SelectionReason reason{SelectionReason::NoCandidate};
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

  // 兼容入口：使用内部 PlannerConfig 和默认物理参数构造 Context。
  AimPlan plan(const std::optional<L3Estimation::TargetState>& target, const L1Sensor::RobotState& robot_state);
  // 主入口：使用本周期 Context 中的 PlannerConfig，对四块装甲板分别
  // 迭代并选择收敛且处于射击窗口内的最佳目标。
  AimPlan plan(
    const std::optional<L3Estimation::TargetState>& target,
    const L1Sensor::RobotState& robot_state,
    const PlannerContext& context);

private:
  PlannerConfig config_; // 当前 Planner 使用的模型和收敛参数
};

}  // namespace L4Planning
