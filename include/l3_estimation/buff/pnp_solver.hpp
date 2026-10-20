#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/buff/types.hpp"
#include "l3_estimation/buff/types.hpp"

#include <Eigen/Geometry>
#include <array>
#include <optional>

namespace L3Estimation::Buff {

struct PnpResult {
  Eigen::Vector3d translation_camera{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation_camera{Eigen::Quaterniond::Identity()};
};

class PnpSolver {
public:
  explicit PnpSolver(const L1Sensor::CameraCalibration& calibration);

  [[nodiscard]] std::optional<PnpResult> solve(
    const L2Perception::Buff::Icon& icon,
    const L2Perception::Buff::Bullseye& bullseye) const;

private:
  L1Sensor::CameraCalibration calibration_;
};

}  // namespace L3Estimation::Buff
