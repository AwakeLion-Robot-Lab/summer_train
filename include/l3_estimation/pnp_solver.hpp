#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <array>
#include <optional>

#include <opencv2/core.hpp>

namespace L3Estimation {

// 单位均为米。
struct ArmorDimensions {
  double small_width = 0.135;
  double large_width = 0.230;
  double height = 0.055;
};

class PnpSolver {
public:
  PnpSolver(
    L1Sensor::CameraCalibration calibration,
    ArmorDimensions dimensions = {});

  // detection.corners 顺序必须为：左上、右上、右下、左下。
  [[nodiscard]] std::optional<ArmorPose> solve(
    const L2Perception::ArmorDetection& detection,
    ArmorSize size) const;

private:
  [[nodiscard]] std::array<cv::Point3f, 4> objectPoints(ArmorSize size) const;

  L1Sensor::CameraCalibration calibration_;
  ArmorDimensions dimensions_;
};

}  // namespace L3Estimation
