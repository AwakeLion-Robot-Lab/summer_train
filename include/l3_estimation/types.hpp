#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor.hpp"

#include <Eigen/Core>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>

namespace L3Estimation {

// L3 中的时间戳统一使用单调时钟，避免系统时间校准造成负时间差。
using TimePoint = std::chrono::steady_clock::time_point;
using ArmorName = L2Perception::ArmorClass;

// 物理装甲板板型只决定 PnP 几何尺寸，车辆类别由 Armor::name 单独表示。
enum class ArmorType : std::uint8_t {
  Small,  // 小装甲板
  Big     // 大装甲板
};

// 识别类别 → 实际板型。场上只有四板车，大装甲板仅英雄使用：平衡步兵已不存在，
// 基地虽然有 Bs/Bb 两个类别但装甲板实物都是小板，所以只有 Hero 走 Big 分支。
//
// 未知类别返回 nullopt，不猜板型：猜错会同时污染 PnP 几何和火控的角度容差。
// L3 的 PnpSolver 和 L5 的 FireDecider 共用这一份映射，不各写一份。
[[nodiscard]] constexpr std::optional<ArmorType> armorTypeOf(ArmorName name) noexcept
{
  switch (name) {
    case ArmorName::Hero:
      return ArmorType::Big;

    case ArmorName::Guard:
    case ArmorName::Engineer:
    case ArmorName::Infantry3:
    case ArmorName::Infantry4:
    case ArmorName::Infantry5:
    case ArmorName::Outpost:
    case ArmorName::BaseSmall:
    case ArmorName::BaseLarge:
      return ArmorType::Small;

    case ArmorName::Unknown:
      break;
  }
  return std::nullopt;
}

// Tracker 的四态生命周期。
enum class TrackState : std::uint8_t {
  Lost,       // 当前没有可用目标
  Detecting,  // 已发现目标，等待连续帧确认
  Tracking,   // 稳定跟踪
  TempLost    // 短时丢失，继续输出预测状态
};

// 相机内参及静态机械外参由 L1 持有，L3 只补充时间同步状态。
struct AimCalibration {
  // 内参、畸变参数以及 camera -> barrel 的静态外参。
  L1Sensor::CameraCalibration camera;
  // 图像曝光时刻与枪管姿态已经完成时间对齐。
  bool time_sync_ok{false};

  // 这里只检查 PnP 所需矩阵是否存在，矩阵数值由 PnpSolver 进一步验证。
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

// L3 对 L2 输出的 Armor 执行单板 PnP 和坐标变换后得到的观测。
struct Armor {
  // 分类信息。name 是车辆类别，type 是实际采用的物理板型。
  ArmorName name{ArmorName::Unknown};
  ArmorType type{ArmorType::Small};
  int class_id{-1};

  // 图像角点顺序固定为左上、右上、右下、左下，单位为 pixel。
  std::array<cv::Point2f, 4> points{};
  // L2 检测得到的四角点几何中心，单位为 pixel。
  cv::Point2f center{};

  // 平移量单位均为 meter。
  Eigen::Vector3d xyz_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};
  // 固定顺序为 [yaw, pitch, roll]，采用 Rz(yaw)Ry(pitch)Rx(roll)。
  Eigen::Vector3d ypr_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_world{Eigen::Vector3d::Zero()};
  // [方位角, 俯仰角, 距离]，角度单位为 radian，距离单位为 meter。
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypd_in_barrel{Eigen::Vector3d::Zero()};

  // 四个角点的二维像素 RMSE，取自 IPPE 在相机系的原始解，与 yaw 优化无关。
  double reprojection_error{std::numeric_limits<double>::infinity()};
  // 检测置信度和四边形像素面积从 L2 原样传入。
  double confidence{0.0};
  double yaw_raw{0.0};
  double area{0.0};
  // 保留旧遥测字段以维持接口兼容。sp_vision 的离散 yaw 搜索不估计标准差，
  // 因而保持无穷；EKF 也不消费这个字段。
  double yaw_sigma{std::numeric_limits<double>::infinity()};

  // 对应原始图像的曝光时刻。
  TimePoint timestamp{};
};

struct ArmorConfig {
  // 装甲板几何尺寸，单位为 meter。
  double small_width{0.135};
  double big_width{0.230};
  double height{0.056};
  // 预留的角点噪声字段；当前离散 yaw 搜索尚未使用。
  double corner_noise_px{1.0};
};

struct TrackerConfig {
  // 从 Detecting 转入 Tracking 所需的连续有效观测帧数。
  int min_detect_count{5};
  // 非 Lost 状态允许的最大相邻帧间隔；超时后重置当前跟踪。
  std::chrono::milliseconds max_frame_interval{100};
  // 临时丢失按连续帧数计数；前哨站允许更长的无观测预测窗口。
  int max_temp_lost_count{15};
  int outpost_max_temp_lost_count{75};
};

// 跨层接口使用的语义别名。
using ArmorObservation = Armor;

}  // namespace L3Estimation
