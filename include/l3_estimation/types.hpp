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
// 对外目标状态固定为九维，元素顺序见 Target::vector()。
using TargetVector = Eigen::Matrix<double, 9, 1>;
using TargetCovariance = Eigen::Matrix<double, 9, 9>;
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

// 不把多个质量条件压成一个 bool，便于分别记录失败原因。
struct ArmorQuality {
  bool pnp_ok{false};           // PnP 求解器返回成功
  bool reprojection_ok{false};  // 像素重投影误差未超过门限
  bool geometry_ok{false};      // 所有物理角点均位于相机前方
  bool finite{false};           // 输出位姿和误差均为有限值

  // 只有全部质量检查通过的观测才能进入目标跟踪器。
  [[nodiscard]] bool valid() const noexcept
  {
    return pnp_ok && reprojection_ok && geometry_ok && finite;
  }
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

  // 四个角点的二维像素 RMSE。
  double reprojection_error{std::numeric_limits<double>::infinity()};
  // 检测置信度和四边形像素面积从 L2 原样传入。
  double confidence{0.0};
  double yaw_raw{0.0};
  double area{0.0};

  ArmorQuality quality;
  // 对应原始图像的曝光时刻。
  TimePoint timestamp{};
};

// 跨层输出使用的九维整车状态：
// [xc, vx, yc, vy, z, vz, yaw, v_yaw, radius]。
struct Target {
  // 当前跟踪车辆和最近一次关联到的物理装甲板编号。
  ArmorName name{ArmorName::Unknown};
  int target_id{-1};
  int armor_id{-1};

  // 旋转中心在世界坐标系中的位置和速度，单位分别为 meter、meter/second。
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  // 整车中心 yaw、角速度以及当前主半径，单位为 radian、radian/second、meter。
  double yaw{0.0};
  double v_yaw{0.0};
  double radius{0.0};

  // 整车几何的其余部分。L4 需要靠这三个字段自行展开全部物理装甲板并外推到
  // 命中时刻，否则只能拿到当前帧的装甲板位置，无法做延迟补偿和选板。
  // second_radius 是四板车奇数板使用的另一组半径（已含 r2-r1），height_diff
  // 是对应的高度差 z2-z1；三板车两者与主半径、0 相同。
  int armor_num{4};
  double second_radius{0.0};
  double height_diff{0.0};

  // 是否已经关联到过 0 号以外的装甲板。为 false 时整车 yaw、第二组半径和
  // 高度差几乎不可观测——只见过一块板的话，其余板的位置完全由初值猜出来。
  // L4 必须据此只瞄当前观测到的那块板，否则等于拿伪造的几何去开火，违反
  // "缺失标定保持缺失"的约定。一旦观测到过就保持为 true。
  bool multi_armor_observed{false};

  // P 与 vector() 使用完全相同的九维元素顺序。
  TargetCovariance P{TargetCovariance::Identity()};
  TrackState track_state{TrackState::Lost};
  TimePoint timestamp{};

  // nis 是最近一次滤波更新的创新统计量；updated 表示本帧使用了观测更新。
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
  // 装甲板几何尺寸，单位为 meter。
  double small_width{0.135};
  double big_width{0.230};
  double height{0.056};
  // 单位分别为 pixel RMSE 和 pixel²。
  double max_reprojection_error{3.0};
  double min_area{20.0};
};

struct TrackerConfig {
  // 从 Detecting 转入 Tracking 所需的连续有效观测帧数。
  int min_detect_count{5};
  // 非 Lost 状态允许的最大相邻帧间隔。
  std::chrono::milliseconds max_frame_interval{100};
  // 临时丢失按连续帧数计数；前哨站允许更长的无观测预测窗口。
  int max_temp_lost_count{15};
  int outpost_max_temp_lost_count{75};

  // 观测是否必须通过 ArmorQuality 的全部门限才能进入滤波。
  //
  // 只有离线回放调试才允许关掉：关掉后门限退化成"single_pnp 成功产出了位姿"，
  // 等于把未经重投影和角点可见性校验的 PnP 结果直接喂给 EKF，发散和跳变都属于
  // 预期内的现象。实机必须保持 true，否则一次坏解就能把整车状态带跑。
  bool require_quality{true};
};

// 跨层接口使用的语义别名。
using ArmorObservation = Armor;
using TargetState = Target;
using TrackStatus = TrackState;

}  // namespace L3Estimation
