#include "l3_estimation/target_estimator.hpp"

namespace L3Estimation {

std::optional<TargetState> TargetEstimator::update(
  const std::vector<L2Perception::Detection>& detections,
  const L1Sensor::RobotState& robot_state,
  std::chrono::steady_clock::time_point timestamp)
{
  (void)detections;
  (void)robot_state;
  (void)timestamp;
  return std::nullopt;
}

}  // namespace L3Estimation
