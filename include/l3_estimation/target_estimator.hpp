#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l2_perception/detector.hpp"
#include "l3_estimation/types.hpp"

#include <chrono>
#include <optional>
#include <vector>

namespace L3Estimation {

class TargetEstimator {
public:
  std::optional<TargetState> update(
    const std::vector<L2Perception::Detection>& detections,
    const L1Sensor::RobotState& robot_state,
    std::chrono::steady_clock::time_point timestamp);
};

}  // namespace L3Estimation
