#pragma once

#include "l2_perception/armor.hpp"

#include <Eigen/Core>

#include <chrono>
#include <cstddef>
#include <numbers>
#include <vector>

#include <opencv2/core.hpp>

namespace L3Estimation {

using TimePoint = std::chrono::steady_clock::time_point;

enum class ArmorSize {
  Small,
  Large
};

enum class TargetModel {
  FourArmorVehicle,
  ThreeArmorOutpost
};

struct TargetModelTraits {
  int armor_count = 4;
  double face_angle_interval_rad = std::numbers::pi / 2.0;
  bool uses_alternating_radius_and_height = true;
};

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

// 每次处理一帧时显式传入时间戳和图像尺寸。
struct FrameContext {
  TimePoint timestamp{};
  cv::Size image_size{};
};

// 单次 IPPE 解出的 armor→camera 位姿，长度单位为米。
struct ArmorPose {
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  double reprojection_error_px = 0.0;
};

enum class TrackerState {
  Lost,
  Detecting,
  Tracking,
  TemporaryLost
};

// 一块二维装甲板经过 PnP、yaw 遍历和坐标变换后的世界系观测。
struct ArmorObservation {
  int robot_id = -1;
  L2Perception::ArmorClass armor_class = L2Perception::ArmorClass::Unknown;
  TargetModel model = TargetModel::FourArmorVehicle;

  Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
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
  double match_cost = 0.0;
  Eigen::Vector4d innovation = Eigen::Vector4d::Zero();
  double nis = 0.0;
  bool nis_valid = false;
  bool accepted = false;
  TrackerState lifecycle_before = TrackerState::Lost;
  TrackerState lifecycle_after = TrackerState::Lost;
};

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
  TimePoint timestamp{};
  TimePoint last_observation_time{};
  bool updated_this_frame = false;
  TargetQualityMetrics quality{};
};

}  // namespace L3Estimation
