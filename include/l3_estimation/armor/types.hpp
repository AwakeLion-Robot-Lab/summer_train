#pragma once

#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <numbers>
#include <limits>
#include <optional>

namespace L3Estimation {

// 车辆类别沿用 L2 的模型类别；板型（Small/Big）只决定 PnP 几何，两者不要混。
using ArmorName = L2Perception::ArmorClass;

enum class ArmorType : std::uint8_t {
  Small,  // 小装甲板
  Big     // 大装甲板
};

// 识别类别 → 实际板型。场上只有四板车，大装甲板仅英雄使用：平衡步兵已不存在，
// 基地虽然有 Bs/Bb 两个类别但装甲板实物都是小板，所以只有 Hero 走 Big 分支。
//
// 未知类别返回 nullopt，不猜板型：猜错会同时污染 PnP 几何和火控的角度容差。
// L3 的 PnpSolver 和 L5 的 FireDecider 共用这一份映射，不各写一份。
constexpr std::optional<ArmorType> armorTypeOf(ArmorName name) noexcept
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

// 识别类别 → 车辆物理装甲板数量。前哨与基地是三板，其余按四板整车模型。
//
// 这一份同时被 Tracker::initializeTarget 的整车初始化和 PnpSolver 的双板配对
// 读取：双板联合 yaw 要求两块相邻板的朝向差恰好 2π/n，n 写错会把整车 yaw 直接
// 拉偏 30 度（前哨按 90 度配对就是这个错）。未知类别返回 nullopt，不猜板数。
// 装甲板绕自身水平轴的后仰角，单位 rad。常规车的板顶向后倾 15 度；前哨站的
// 三块板反过来向前倾，所以取负。
//
// 这个量有三个用处，必须同一个来源：PnP 的物点姿态、叠加层画法向箭头、
// 火控算竖直命中窗口。分头写死会在改板型时漏掉其中一处。
constexpr double armorPitchOf(ArmorName name) noexcept
{
  constexpr double kTilt = 15.0 * std::numbers::pi / 180.0;
  return name == ArmorName::Outpost ? -kTilt : kTilt;
}

constexpr std::optional<int> armorCountOf(ArmorName name) noexcept
{
  switch (name) {
    case ArmorName::Outpost:
    case ArmorName::BaseSmall:
    case ArmorName::BaseLarge:
      return 3;

    case ArmorName::Guard:
    case ArmorName::Engineer:
    case ArmorName::Hero:
    case ArmorName::Infantry3:
    case ArmorName::Infantry4:
    case ArmorName::Infantry5:
      return 4;

    case ArmorName::Unknown:
      break;
  }
  return std::nullopt;
}

// Tracker 的四态生命周期。
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

// 整车 EKF 的过程噪声与观测噪声。这两组是靠回放标定的主要旋钮，所以出到
// 配置；半径物理范围、前哨固定转速这类物理常量仍留在代码里。
struct TargetConfig {
  // 过程噪声强度。平移与高度用 translation，整车 yaw 用 rotation。
  double q_translation{100.0};
  double q_rotation{400.0};
  // 前哨站转速固定、轨迹规整，过程噪声显著更小。
  double outpost_q_translation{10.0};
  double outpost_q_rotation{0.1};

  // 观测噪声，观测量为 [方位角, 俯仰角, 距离, 装甲板 yaw]。
  // 方位角/俯仰角取常量方差。
  double angle_variance{4e-3};
  // 距离方差 = factor * d^2 * (1 + delta_angle^2)。单板 PnP 的深度误差
  // 大致正比于距离平方（板在像素上的张角 ∝ 1/d），斜视时进一步变差。
  // 默认 0.0625 使 4 m 正视处的方差等于 1.0 m^2。
  double distance_variance_factor{0.0625};
  // 板 yaw 方差 = base + log1p(d) / distance_divisor。
  double armor_yaw_variance_base{9e-2};
  double armor_yaw_distance_divisor{200.0};
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
