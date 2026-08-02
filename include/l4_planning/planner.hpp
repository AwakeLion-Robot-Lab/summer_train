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

struct GimbalExtrinsics {
  // gimbal frame -> barrel frame
  Eigen::Isometry3d T_barrel_gimbal{Eigen::Isometry3d::Identity()};
};

struct PlannerContext {
  TimePoint planning_time{};
  LatencyConfig latency;
  Eigen::Vector3d gimbal_center_world{Eigen::Vector3d::Zero()};
  GimbalExtrinsics gimbal_extrinsics;
  double gravity{9.80665};
  PlannerConfig config;
  ArmorScoreWeights armor_score_weights;
};

struct ArmorCandidate {
  ArmorPose armor;
  BallisticSolution ballistic;
  TimePoint impact_time{};
  double delta_angle{0.0};
  int iteration_count{0};
  double fly_time_error{0.0};
  double position_error{0.0};
  bool converged{false};
  bool within_firing_window{false};
  bool valid{false};
  ArmorScore score;
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
  AimPlan plan(const std::optional<L3Estimation::TargetState>& target, const L1Sensor::RobotState& robot_state);
};

}  // namespace L4Planning
