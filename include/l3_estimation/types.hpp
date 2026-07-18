#pragma once

#include "l2_perception/armor.hpp"

#include <Eigen/Core>

#include <chrono>

#include <opencv2/core.hpp>

namespace L3Estimation {

using TimePoint = std::chrono::steady_clock::time_point;

enum class ArmorSize {
  Small,
  Large
};

// 装甲板局部坐标系到 OpenCV 相机坐标系的原始 PnP 位姿。
// 装甲板坐标：x 沿法向、y 向左、z 向上；tvec 单位为米。
struct ArmorPose {
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
};

// 完成 PnP、世界坐标转换和 yaw 重投影优化后的单板观测。
struct ArmorObservation {
  int robot_id = -1;
  L2Perception::ArmorClass armor_class = L2Perception::ArmorClass::Unknown;

  Eigen::Vector3d position_world{};
  double yaw_raw_world = 0.0;
  double yaw_world = 0.0;

  float confidence = 0.0F;
  double reprojection_error_px = 0.0;
  TimePoint timestamp{};
};

enum class TrackerState {
  Lost,
  Detecting,
  Tracking,
  TemporaryLost
};

// L3 对 L4 发布的整车滤波状态。
struct VehicleState {
  int robot_id = -1;

  Eigen::Vector3d center_world{};
  Eigen::Vector3d velocity_world{};

  double yaw = 0.0;
  double yaw_rate = 0.0;
  double radius = 0.0;
  int armor_count = 4;

  TrackerState tracker_state = TrackerState::Lost;
  double tracking_quality = 0.0;
  TimePoint timestamp{};
};

}  // namespace L3Estimation
