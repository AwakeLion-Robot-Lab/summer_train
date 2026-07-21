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

// 整车 EKF 的固定状态顺序。covariance 的行列必须严格使用这一顺序：
// [xc, vx, yc, vy, zc, vz, yaw, yaw_rate, r1, r2-r1, z2-z1]。
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

// L3 对 L4 发布的目标车辆状态。
//
// 坐标约定与 tongjiceshi 一致：OpenCV 相机系为 x右、y下、z前；经过
// camera->gimbal->world 后，世界系为 x前、y左、z上，原点位于云台旋转中心。
// 所有长度使用 m，速度使用 m/s，角度使用 rad，角速度使用 rad/s。
struct TargetState {
  int robot_id = -1;

  // 车辆旋转中心的水平位置；z 是第 1 组装甲板的参考高度，不一定是车辆
  // 几何中心高度。
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();

  // 0 号装甲板局部 +x 法线在世界系中的方向。yaw=0 指向世界系 +x，
  // 从 +x 转向 +y（向左转）为正，发布前应归一化到 (-pi, pi]。
  double yaw = 0.0;
  double yaw_rate = 0.0;

  // r1、r2-r1、z2-z1。四装甲板模型中 0/2 号面使用 r1/z1，
  // 1/3 号面使用 r2/z2。
  double radius = 0.0;
  double radius_offset = 0.0;
  double height_offset = 0.0;

  StateCovariance covariance = StateCovariance::Identity();

  // 状态实际对应的预测/更新时刻，而不是发送给 L4 的时刻。
  TimePoint timestamp{};
};

}  // namespace L3Estimation
