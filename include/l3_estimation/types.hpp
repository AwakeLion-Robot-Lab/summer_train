#pragma once

#include "l2_perception/armor.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <opencv2/core.hpp>

namespace L3Estimation {

// 单调时钟时间戳，回放与实机共用。
using TimePoint = std::chrono::steady_clock::time_point;

// 装甲板尺寸：小/大，对应不同板宽。
enum class ArmorSize {
  Small,
  Large
};

// 目标模型：普通四板车 / 三板前哨站。
enum class TargetModel {
  FourArmorVehicle,
  ThreeArmorOutpost
};

// 目标模型的几何特征：面数、面间隔、是否交替使用第二组半径/高度。
struct TargetModelTraits {
  int armor_count = 4;
  double face_angle_interval_rad = std::numbers::pi / 2.0;
  bool uses_alternating_radius_and_height = true;
};

// 按模型返回几何特征。
[[nodiscard]] constexpr TargetModelTraits targetModelTraits(
  TargetModel model) noexcept
{
  if (model == TargetModel::ThreeArmorOutpost) {
    return {
      .armor_count = 3,
      .face_angle_interval_rad = 2.0 * std::numbers::pi / 3.0,
      .uses_alternating_radius_and_height = false};
  }
  return {};
}

// 四装甲模型俯视图中，相邻装甲中心连成的四边形具有两种内角。
// r1/r2 是两组相对装甲到旋转中心的半径，返回较小内角，范围 (0, pi/2]。
[[nodiscard]] inline double fourArmorMinimumCornerAngle(
  double first_radius,
  double second_radius) noexcept
{
  if (!std::isfinite(first_radius)
      || !std::isfinite(second_radius)
      || first_radius <= 0.0
      || second_radius <= 0.0) {
    return 0.0;
  }
  return 2.0 * std::atan2(
    std::min(first_radius, second_radius),
    std::max(first_radius, second_radius));
}

// 每次处理一帧时显式传入时间戳和图像尺寸。
struct FrameContext {
  TimePoint timestamp{};
  cv::Size image_size{};
};

// 单个 IPPE 候选解出的 armor→camera 位姿，长度单位为米。
struct ArmorPose {
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  double reprojection_error_px = 0.0;
  // solvePnPGeneric 返回的候选下标（0 或 1），用于诊断和确定性平手选择。
  int ippe_candidate_index = 0;
};

// 单假设目标生命周期。
enum class TrackerState {
  Lost,
  Detecting,
  Tracking,
  TemporaryLost
};

// 一块二维装甲板经过 PnP、yaw 遍历和坐标变换后的世界系观测。
struct ArmorObservation {
  // 对应本帧 ArmorDetection 数组下标，供回放工具把 L3 信息画回检测框。
  std::size_t source_detection_index = 0;
  int robot_id = -1;
  L2Perception::ArmorClass armor_class = L2Perception::ArmorClass::Unknown;
  TargetModel model = TargetModel::FourArmorVehicle;

  Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
  // RPY 顺序固定为 [roll, pitch, yaw]，单位 rad。
  Eigen::Vector3d rpy_raw_world = Eigen::Vector3d::Zero();
  // 基线姿态约束为 Rz(yaw)Ry(configured_pitch)，因此 roll 固定为 0。
  Eigen::Vector3d rpy_constrained_world = Eigen::Vector3d::Zero();
  double yaw_raw_world = 0.0;
  double yaw_world = 0.0;

  float confidence = 0.0F;
  double pnp_reprojection_error_px = 0.0;
  double raw_yaw_reprojection_error_px = 0.0;
  double optimized_reprojection_error_px = 0.0;
  TimePoint timestamp{};
};

// 基线版本每块观测只记录一次简单物理面匹配结果。
struct AssociationDiagnostic {
  std::size_t observation_index = 0;
  int robot_id = -1;
  int associated_face_id = -1;
  double position_error_m = 0.0;
  double yaw_error_rad = 0.0;
  double implied_radius_m = 0.0;
  double radius_error_m = 0.0;
  double minimum_corner_angle_rad = 0.0;
  double match_cost = 0.0;
  Eigen::Vector4d innovation = Eigen::Vector4d::Zero();
  double nis = 0.0;
  bool nis_valid = false;
  // 观测-预测残差 [x,y,z,yaw] 与标准 NIS；第一版只记录，不参与拒绝。
  bool accepted = false;
  TrackerState lifecycle_before = TrackerState::Lost;
  TrackerState lifecycle_after = TrackerState::Lost;
};

// 每次帧更新后汇总的质量指标。
struct TargetQualityMetrics {
  float mean_detection_confidence = 0.0F;
  double mean_reprojection_error_px = 0.0;
  double last_nis = 0.0;
  bool nis_valid = false;
  std::size_t accepted_observation_count = 0;
  int associated_face_id = -1;
  Eigen::Vector4d innovation = Eigen::Vector4d::Zero();
  TrackerState lifecycle_before = TrackerState::Lost;
  TrackerState lifecycle_after = TrackerState::Lost;
};

// 整车 EKF 固定使用这一状态顺序。
enum StateIndex : int {
  XC = 0,
  VX = 1,
  YC = 2,
  VY = 3,
  ZC = 4,
  VZ = 5,
  YAW = 6,
  YAW_RATE = 7,
  RADIUS = 8,
  RADIUS_OFFSET = 9,
  HEIGHT_OFFSET = 10,
  STATE_DIM = 11
};

using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using StateCovariance = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;

// L3 发布的世界系整车状态。
struct TargetState {
  int robot_id = -1;
  TargetModel model = TargetModel::FourArmorVehicle;
  int armor_count = targetModelTraits(model).armor_count;
  TrackerState tracker_state = TrackerState::Lost;

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  double yaw = 0.0;
  double yaw_rate = 0.0;
  double radius = 0.0;
  double radius_offset = 0.0;
  double height_offset = 0.0;

  StateCovariance covariance = StateCovariance::Identity();
  // filter 时间戳 / 最近一次观测时间 / 本帧是否有观测参与更新。
  TimePoint timestamp{};
  TimePoint last_observation_time{};
  bool updated_this_frame = false;
  TargetQualityMetrics quality{};
};

}  // namespace L3Estimation
