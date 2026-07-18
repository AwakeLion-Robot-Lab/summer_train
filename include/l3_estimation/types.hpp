#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>

namespace L3Estimation {

using TimePoint = std::chrono::steady_clock::time_point;
using TargetVector = Eigen::Matrix<double, 9, 1>;
using TargetCovariance = Eigen::Matrix<double, 9, 9>;
using ArmorCovariance = Eigen::Matrix4d;
using ArmorName = L2Perception::ArmorClass;

// 物理装甲板尺寸。Armor::name 表示车辆编号，type 只表示板型。
enum class ArmorType : std::uint8_t {
  Small,
  Big
};

// 第一版只使用 Single；DoubleYaw 为后续双板联合 yaw 观测预留。
enum class ObservationMode : std::uint8_t {
  Single,
  DoubleYaw
};

enum class TrackState : std::uint8_t {
  Lost,
  Detecting,
  Tracking,
  TempLost
};

// 不把多个质量条件压成一个 bool，便于 L6 分别记录失败原因。
struct ArmorQuality {
  bool pnp_ok{false};
  bool covariance_ok{false};
  bool reprojection_ok{false};
  bool geometry_ok{false};
  bool finite{false};
  // 两个可用 IPPE 解的像素误差过于接近时，rpy_in_world.z() 不可信。
  bool yaw_ambiguous{false};

  [[nodiscard]] bool valid() const noexcept
  {
    return pnp_ok && covariance_ok && reprojection_ok && geometry_ok && finite;
  }
};

// L3 只补充观测协方差和时间同步状态；相机内参及静态机械外参
// 由 L1 持有。
struct AimCalibration {
  L1Sensor::CameraCalibration camera;

  // 对应观测 [x, y, z, yaw] 的共享标定和姿态同步不确定度。
  ArmorCovariance R_calibration{ArmorCovariance::Zero()};
  ArmorCovariance R_pose_sync{ArmorCovariance::Zero()};

  bool time_sync_ok{false};

  [[nodiscard]] bool intrinsicsOk() const noexcept
  {
    return !camera.camera_matrix.empty() &&
           !camera.distortion_coefficients.empty();
  }

  [[nodiscard]] bool trackingReady() const noexcept
  {
    return intrinsicsOk() && camera.barrelExtrinsicsReady() && time_sync_ok;
  }

  [[nodiscard]] bool fireReady() const noexcept
  {
    return trackingReady();
  }
};

// 一帧图像进入 L3 时的附加信息，姿态必须对应图像曝光时刻。
struct FrameInfo {
  TimePoint timestamp{};
  Eigen::Quaterniond q_world_barrel{Eigen::Quaterniond::Identity()};
  bool barrel_pose_ok{false};
};

// L3 对 L2 输出的 Armor 执行单板 PnP 和坐标变换后得到的观测。
// R 的变量顺序固定为 [x_world, y_world, z_world, yaw_world]。
struct Armor {
  ArmorName name{ArmorName::Unknown};
  ArmorType type{ArmorType::Small};
  int class_id{-1};
  int id{-1};  // 与整车物理装甲板关联后的 0~3 编号

  std::array<cv::Point2f, 4> points{};

  Eigen::Vector3d xyz_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};
  // 固定顺序为 [roll, pitch, yaw]，采用 Rz(yaw)Ry(pitch)Rx(roll)。
  Eigen::Vector3d rpy_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rpy_in_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};

  // 四个角点的二维像素 RMSE。
  double reprojection_error{std::numeric_limits<double>::infinity()};
  // 第二个几何有效 IPPE 候选的像素 RMSE；不存在时为无穷大。
  double second_reprojection_error{
    std::numeric_limits<double>::infinity()};
  double confidence{0.0};
  double area{0.0};
  double facing{0.0};

  ObservationMode mode{ObservationMode::Single};
  ArmorCovariance R{ArmorCovariance::Identity()};
  ArmorQuality quality;
  TimePoint timestamp{};
};

// 普通四板车辆的九维整车状态：
// [xc, vx, yc, vy, z, vz, yaw, v_yaw, radius]。
struct Target {
  ArmorName name{ArmorName::Unknown};
  int target_id{-1};
  int armor_id{-1};

  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double yaw{0.0};
  double v_yaw{0.0};
  double radius{0.0};

  TargetCovariance P{TargetCovariance::Identity()};
  TrackState track_state{TrackState::Lost};
  TimePoint timestamp{};

  double nis{0.0};
  bool updated{false};

  [[nodiscard]] TargetVector vector() const noexcept
  {
    TargetVector x;
    x << position.x(), velocity.x(), position.y(), velocity.y(), position.z(),
      velocity.z(), yaw, v_yaw, radius;
    return x;
  }
};

struct ArmorConfig {
  double small_width{0.135};
  double big_width{0.225};
  double height{0.055};
  double corner_noise{0.8};  // pixel standard deviation
  double max_reprojection_error{3.0};
  double ambiguity_error_gap{0.25};  // pixel RMSE
  double ambiguity_error_ratio{1.2};
  double min_area{20.0};
};

struct TrackerConfig {
  int min_detect_count{3};
  std::chrono::milliseconds max_temp_lost{100};
  double default_radius{0.22};
  double min_radius{0.12};
  double max_radius{0.45};
  double nis_gate{18.5};
};

// 与设计文档中的名称兼容，业务代码优先使用上面的短名称。
using ArmorObservation = Armor;
using TargetState = Target;
using TrackStatus = TrackState;

}  // namespace L3Estimation
