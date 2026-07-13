#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l2_perception/detector.hpp"

#include <chrono>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace L3Estimation {

struct TargetState {
  cv::Point3d position;
  cv::Point3d velocity;
  int target_id = -1;
  std::chrono::steady_clock::time_point timestamp{};
};

class TargetEstimator {
public:
  std::optional<TargetState> update(
    const std::vector<L2Perception::Detection>& detections,
    const L1Sensor::RobotState& robot_state,
    std::chrono::steady_clock::time_point timestamp);
};

}  // namespace L3Estimation
