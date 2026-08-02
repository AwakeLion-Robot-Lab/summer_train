#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <array>
#include <optional>

namespace L3Estimation {

// 单位均为米。
struct ArmorDimensions {
  double small_width = 0.135;
  double large_width = 0.230;
  double height = 0.055;
};

struct PnpSolverConfig {
  double minimum_corner_area_px = 1.0;
  double minimum_distance_m = 0.05;
  double maximum_distance_m = 20.0;
  double maximum_reprojection_error_px = 8.0;
};

// 基线版本只调用一次 SOLVEPNP_IPPE，并返回一个通过检查的位姿。
class PnpSolver {
public:
  PnpSolver(
    L1Sensor::CameraCalibration calibration,
    ArmorDimensions dimensions = {},
    PnpSolverConfig config = {});

  // detection.corners 顺序必须为：左上、右上、右下、左下。
  [[nodiscard]] std::optional<ArmorPose> solve(
    const L2Perception::ArmorDetection& detection,
    ArmorSize size) const;

private:
  [[nodiscard]] std::array<cv::Point3f, 4> objectPoints(
    ArmorSize size) const;

  L1Sensor::CameraCalibration calibration_;
  ArmorDimensions dimensions_;
  PnpSolverConfig config_;
};

}  // namespace L3Estimation
