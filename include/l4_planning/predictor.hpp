#pragma once

#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Core>

#include <vector>

namespace L4Planning {

enum class ArmorType {
  Small,
  Large
};

struct PredictionRequest {
  L3Estimation::TargetState target;
  TimePoint target_time{};
};

struct ArmorPose {
  int robot_id{-1};
  int armor_id{-1};
  ArmorType armor_type{ArmorType::Small};

  Eigen::Vector3d position_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_world{Eigen::Vector3d::Zero()};
  double yaw_world{0.0};

  TimePoint timestamp{};
  bool valid{false};
};

struct PredictionResult {
  L3Estimation::TargetState predicted_vehicle;
  std::vector<ArmorPose> armor_candidates;
  bool valid{false};
};

class Predictor {
public:
  L3Estimation::TargetState predict(const L3Estimation::TargetState& target, double dt) const;
  [[nodiscard]] PredictionResult predict(const PredictionRequest& request) const;
};

}  // namespace L4Planning
