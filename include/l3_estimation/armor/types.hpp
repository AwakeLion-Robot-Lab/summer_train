#pragma once

#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <numbers>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace L3Estimation {

/// 车辆类别沿用 L2 的模型类别；板型（Small/Big）只决定 PnP 几何，两者不要混。
using ArmorName = L2Perception::ArmorClass;

/**
 * @brief 装甲板的物理板型，只决定 PnP 几何
 */
enum class ArmorType : std::uint8_t {
  Small,  ///< 小装甲板
  Big     ///< 大装甲板
};

/**
 * @brief 识别类别 → 实际板型
 *
 * 映射本身在 L2 的 isLargeArmor，这里只换成 L3 的枚举，不另写一份。
 * L3 的 PnpSolver 和 L5 的 FireDecider 共用这一份映射。
 *
 * @param name 车辆类别
 * @return 板型；未知类别返回 nullopt，不猜板型：猜错会同时污染 PnP 几何和
 *         火控的角度容差
 */
constexpr std::optional<ArmorType> armorTypeOf(ArmorName name) noexcept
{
  const std::optional<bool> large = L2Perception::isLargeArmor(name);
  if (!large) {
    return std::nullopt;
  }
  return *large ? ArmorType::Big : ArmorType::Small;
}

/**
 * @brief 装甲板绕自身水平轴的后仰角
 *
 * 常规车的板顶向后倾 15 度；前哨站的三块板反过来向前倾，所以取负。
 *
 * 这个量有三个用处，必须同一个来源：PnP 的物点姿态、叠加层画法向箭头、
 * 火控算竖直命中窗口。分头写死会在改板型时漏掉其中一处。
 *
 * @param name 车辆类别
 * @return 后仰角，rad
 */
constexpr double armorPitchOf(ArmorName name) noexcept
{
  constexpr double kTilt = 15.0 * std::numbers::pi / 180.0;
  return name == ArmorName::Outpost ? -kTilt : kTilt;
}

/**
 * @brief 识别类别 → 车辆物理装甲板数量
 *
 * 前哨与基地是三板，其余按四板整车模型。相邻板朝向差恰好 2π/n，n 写错会把
 * 整车 yaw 直接拉偏 30 度。
 *
 * @param name 车辆类别
 * @return 板数；未知类别返回 nullopt，不猜板数
 */
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

/**
 * @brief L3 的单块装甲板观测：L2 的检测字段加上 PnP 解出的位姿
 */
struct Armor {
  /// 车辆类别。PnP 失败时置为 Unknown。
  ArmorName name{ArmorName::Unknown};

  /// 实际采用的物理板型
  ArmorType type{ArmorType::Small};

  /// L2 的类别编号
  int class_id{-1};

  /// L2 判出的颜色
  L2Perception::ArmorColor color{L2Perception::ArmorColor::Unknown};

  /// 图像角点顺序固定为左上、右上、右下、左下，单位为 pixel。
  std::array<cv::Point2f, 4> points{};

  /// L2 检测得到的四角点几何中心，单位为 pixel。
  cv::Point2f center{};

  /// 板心在相机系的位置，meter。枪管系只是 camera -> world 的中间量，没有下游
  /// 读它，就不落成字段，免得 PnP 失败时留着上一帧的残值。
  Eigen::Vector3d xyz_in_camera{Eigen::Vector3d::Zero()};

  /// 板心在世界系的位置，meter
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};

  /// 相机系姿态，固定顺序为 [yaw, pitch, roll]，采用 Rz(yaw)Ry(pitch)Rx(roll)。
  Eigen::Vector3d ypr_in_camera{Eigen::Vector3d::Zero()};

  /// 世界系姿态，[yaw, pitch, roll]
  Eigen::Vector3d ypr_in_world{Eigen::Vector3d::Zero()};

  /// [方位角, 俯仰角, 距离]，角度单位为 radian，距离单位为 meter。
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};

  /// 四个角点的二维像素 RMSE，取自 IPPE 在相机系的原始解，与 yaw 优化无关。
  double reprojection_error{std::numeric_limits<double>::infinity()};

  /// 检测置信度，从 L2 原样传入
  double confidence{0.0};

  /// IPPE 原始解的世界系 yaw，yaw 搜索之前的值，rad
  double yaw_raw{0.0};

  /// 四边形像素面积，从 L2 原样传入
  double area{0.0};
};

/**
 * @brief 装甲板几何尺寸，单位为 meter
 */
struct ArmorConfig {
  /// 小板灯条间距
  double small_width{0.135};

  /// 大板灯条间距
  double big_width{0.230};

  /// 灯条高度
  double height{0.056};
};

/// 关联的产物，关联器产出、滤波器消费。放在这里而不是关联器头文件里，是为了让
/// EskfTarget::update 能只依赖类型、不依赖关联怎么做出来的。
///
/// (整车物理板编号, 完整装甲板)。
using MatchedArmor = std::pair<int, Armor>;

/// (整车物理板编号, 是否左灯, 侧边灯条)。灯条还要记左右，因为一根侧边灯条只对应半块板。
using MatchedLight = std::tuple<int, bool, L2Perception::Light>;

}  // namespace L3Estimation
