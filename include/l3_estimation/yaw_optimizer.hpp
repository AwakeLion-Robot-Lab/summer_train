#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Geometry>

#include <array>
#include <optional>

namespace L3Estimation {

// 连续 yaw 优化的迭代与收敛参数。
struct YawOptimizationConfig {
  bool enabled = true;
  int max_iterations = 12;
  // 数值雅可比有限差分步长，仅用于 yaw 分量，单位 rad。
  double parameter_step = 1.0e-4;
  // 参数更新量绝对值上限（yaw，rad）。
  double convergence_tolerance = 1.0e-6;
};

struct YawOptimizationResult {
  // 世界系原始 PnP 姿态，顺序固定为 [roll, pitch, yaw]，单位 rad。
  Eigen::Vector3d rpy_raw_world = Eigen::Vector3d::Zero();
  double yaw_raw_world = 0.0;
  double yaw_optimized_world = 0.0;
  // armor 中心在相机系中的位置，单位 m；当前保持 PnP 原值，
  // 位置精化随第 5 项（阻尼/Huber）接入后再改为优化值。
  cv::Vec3d tvec_optimized_camera{};
  double pnp_reprojection_error_px = 0.0;
  double raw_yaw_reprojection_error_px = 0.0;
  double optimized_reprojection_error_px = 0.0;
  // 残差求值次数（raw 评估固定为 1）。
  int evaluation_count = 0;
};

// 固定装甲 pitch 和 PnP 位置，连续优化 yaw 的重投影误差；也提供 raw 评估。
class YawOptimizer {
public:
  YawOptimizer(
    L1Sensor::CameraCalibration calibration,
    ArmorDimensions dimensions,
    YawOptimizationConfig config);

  // 只做 raw 评估（世界系 raw yaw 与对应重投影误差），不遍历 yaw，
  // 用于选解前廉价比较不同 IPPE 候选。
  [[nodiscard]] std::optional<YawOptimizationResult> raw(
    const L2Perception::ArmorDetection& detection,
    ArmorSize size,
    const ArmorPose& pose,
    const Eigen::Quaterniond& R_world_barrel,
    double configured_armor_pitch_rad) const;

  [[nodiscard]] std::optional<YawOptimizationResult> optimize(
    const L2Perception::ArmorDetection& detection,
    ArmorSize size,
    const ArmorPose& pose,
    const Eigen::Quaterniond& R_world_barrel,
    double configured_armor_pitch_rad) const;

private:
  using ResidualVector = Eigen::Matrix<double, 8, 1>;

  // 按装甲尺寸生成 armor 局部系四角点。
  [[nodiscard]] std::array<cv::Point3d, 4> objectPoints(
    ArmorSize size) const;

  // 校验输入并计算 raw yaw 与对应重投影误差（不做搜索）。
  [[nodiscard]] std::optional<YawOptimizationResult> evaluateRaw(
    const L2Perception::ArmorDetection& detection,
    ArmorSize size,
    const ArmorPose& pose,
    const Eigen::Quaterniond& R_world_barrel,
    double configured_armor_pitch_rad) const;

  // 固定 pitch 模型下，给定 θ=[tx,ty,tz,yaw] 求四角点 8 维像素残差。
  [[nodiscard]] std::optional<ResidualVector> reprojectionResidual(
    const Eigen::Vector4d& theta,
    double configured_armor_pitch_rad,
    const Eigen::Matrix3d& R_camera_world,
    const std::array<cv::Point3d, 4>& object_points,
    const L2Perception::ArmorDetection& detection) const;

  // 残差向量转四角点 RMSE（px）。
  [[nodiscard]] static double residualRmse(
    const ResidualVector& residual) noexcept;

  L1Sensor::CameraCalibration calibration_;
  ArmorDimensions dimensions_;
  YawOptimizationConfig config_;
  Eigen::Matrix3d R_barrel_camera_ = Eigen::Matrix3d::Identity();
};

}  // namespace L3Estimation
