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
