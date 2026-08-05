#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Geometry>

#include <array>
#include <optional>

namespace L3Estimation {

struct YawSearchConfig {
  bool enabled = true;
  double search_half_range_rad = 1.2217304763960306;  // 70°
  double search_step_rad = 0.0174532925199433;        // 1°
};

struct YawOptimizationResult {
  // 世界系原始 PnP 姿态，顺序固定为 [roll, pitch, yaw]，单位 rad。
  Eigen::Vector3d rpy_raw_world = Eigen::Vector3d::Zero();
  double yaw_raw_world = 0.0;
  double yaw_optimized_world = 0.0;
  double pnp_reprojection_error_px = 0.0;
  double raw_yaw_reprojection_error_px = 0.0;
  double optimized_reprojection_error_px = 0.0;
  int evaluated_yaw_count = 0;
};

// 基线版本固定 PnP 位置和装甲 pitch，只遍历世界系 yaw。
class YawOptimizer {
public:
  YawOptimizer(
    L1Sensor::CameraCalibration calibration,
    ArmorDimensions dimensions,
    YawSearchConfig config);

  [[nodiscard]] std::optional<YawOptimizationResult> optimize(
    const L2Perception::ArmorDetection& detection,
    ArmorSize size,
    const ArmorPose& pose,
    const Eigen::Quaterniond& R_world_barrel,
    double configured_armor_pitch_rad) const;

private:
  [[nodiscard]] std::array<cv::Point3d, 4> objectPoints(
    ArmorSize size) const;

  L1Sensor::CameraCalibration calibration_;
  ArmorDimensions dimensions_;
  YawSearchConfig config_;
  Eigen::Matrix3d R_barrel_camera_ = Eigen::Matrix3d::Identity();
};

}  // namespace L3Estimation
