#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <array>
#include <optional>
#include <vector>

namespace L3Estimation {

// 单位均为米。
struct ArmorDimensions {
  double small_width = 0.135;
  double large_width = 0.230;
  double height = 0.055;
};

struct PnpSolverConfig {
  // 开启时保留 solvePnPGeneric 的两个 IPPE 候选；关闭时恢复旧 solvePnP 单候选，
  // 用于同数据 A/B 对比。
  bool enable_ippe_dual_candidates = true;
  // 已确认目标（Tracking/TemporaryLost）用预测 face yaw 选择 IPPE 候选；
  // 关闭时始终按 yaw 优化误差选解，用于同数据 A/B 对比。
  bool enable_predicted_face_yaw_selection = true;
  // PnP 输入与几何门限：四边形最小面积、距离范围和重投影 RMSE 上限。
  double minimum_corner_area_px = 1.0;
  double minimum_distance_m = 0.05;
  double maximum_distance_m = 20.0;
  double maximum_reprojection_error_px = 8.0;
};

// IPPE PnP：返回 0~2 个通过检查的候选位姿（开启双候选时）。
class PnpSolver {
public:
  PnpSolver(
    L1Sensor::CameraCalibration calibration,
    ArmorDimensions dimensions = {},
    PnpSolverConfig config = {});

  // detection.corners 顺序必须为：左上、右上、右下、左下。
  [[nodiscard]] std::vector<ArmorPose> solve(
    const L2Perception::ArmorDetection& detection,
    ArmorSize size) const;

private:
  // 按装甲尺寸生成 armor 局部系四角点。
  [[nodiscard]] std::array<cv::Point3f, 4> objectPoints(
    ArmorSize size) const;

  // 对单个 IPPE 候选做正深度、距离和重投影检查。
  [[nodiscard]] std::optional<ArmorPose> makeCandidate(
    int candidate_index,
    const cv::Vec3d& rvec,
    const cv::Vec3d& tvec,
    const L2Perception::ArmorDetection& detection,
    const std::array<cv::Point3f, 4>& object_points) const;

  L1Sensor::CameraCalibration calibration_;
  ArmorDimensions dimensions_;
  PnpSolverConfig config_;
};

}  // namespace L3Estimation
