#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core/types.hpp>

namespace L2Perception {

/**
 * @brief 视觉判出的装甲板 / 灯条颜色
 *
 * 与 L1Sensor::EnemyColor 是两个来源，任一侧为 Unknown 都不算有效目标。
 */
enum class ArmorColor {
  Red,
  Blue,
  Unknown
};

/**
 * @brief 9 个车辆类别
 *
 * 数值是跨层约定的 class_id（G、1、2、3、4、5、O、Bs、Bb），与整板模型类别
 * 字段 argmax 的下标一致；Armor 里保留 class_id 方便记录，需要语义时调用
 * armorClassFromId。
 */
enum class ArmorClass : int {
  Guard = 0,      ///< G，哨兵
  Hero = 1,       ///< 1，英雄
  Engineer = 2,   ///< 2，工程
  Infantry3 = 3,  ///< 3，步兵三号
  Infantry4 = 4,  ///< 4，步兵四号
  Infantry5 = 5,  ///< 5，步兵五号
  Outpost = 6,    ///< O，前哨站
  BaseSmall = 7,  ///< Bs，小装甲基地
  BaseLarge = 8,  ///< Bb，大装甲基地
  Unknown = -1    ///< 未知类别
};

/**
 * @brief 类别编号 → 车辆类别
 * @param class_id 模型类别 argmax 的下标
 * @return 对应的车辆类别；越界时为 Unknown
 */
constexpr ArmorClass armorClassFromId(int class_id) noexcept
{
  return class_id >= static_cast<int>(ArmorClass::Guard)
             && class_id <= static_cast<int>(ArmorClass::BaseLarge)
           ? static_cast<ArmorClass>(class_id)
           : ArmorClass::Unknown;
}

/**
 * @brief 识别类别 → 是否大装甲板
 *
 * 场上只有四板车，大装甲板仅英雄使用：平衡步兵已不存在，基地虽然有 Bs/Bb 两个
 * 类别但装甲板实物都是小板。
 *
 * L3 的 armorTypeOf 由它派生，PnP 和端点观测的板宽都按这一份映射取。
 *
 * @param armor_class 车辆类别
 * @return 大板为 true、小板为 false；未知类别返回 nullopt，不猜板型
 */
constexpr std::optional<bool> isLargeArmor(ArmorClass armor_class) noexcept
{
  switch (armor_class) {
    case ArmorClass::Hero:
      return true;

    case ArmorClass::Guard:
    case ArmorClass::Engineer:
    case ArmorClass::Infantry3:
    case ArmorClass::Infantry4:
    case ArmorClass::Infantry5:
    case ArmorClass::Outpost:
    case ArmorClass::BaseSmall:
    case ArmorClass::BaseLarge:
      return false;

    case ArmorClass::Unknown:
      break;
  }
  return std::nullopt;
}

/**
 * @brief 整板网络检出的装甲板
 *
 * 这里不带三维位姿：PnP 是 L3 的事，位姿只存在 L3Estimation::Armor 上。
 * L2 再挂一份的话，两份迟早会有一份是陈的。
 */
struct Armor {
  /// 顺序固定为：左上、右上、右下、左下；PnP 必须沿用同一顺序。
  /// 左上/左下是左灯条的上下端点，右上/右下是右灯条的，L3 按此拆成两根灯条。
  std::array<cv::Point2f, 4> corners{};

  /// 四个角点的几何中心。
  cv::Point2f center{};

  /// 车辆类别编号，见 ArmorClass，即整板网络的类别 argmax。
  int class_id{-1};

  /// 网络判出的颜色
  ArmorColor color{ArmorColor::Unknown};

  /// 网络置信度（yolov5 为 sigmoid 后的 objectness）。
  float confidence{0.0F};
};

/**
 * @brief 检出的单根灯条
 *
 * 坐标都在原图像素系，由 searchLights 的剖面搜索给出。它不带车辆编号，只有
 * 端点观测要用的上下端点和几何量；属于哪块装甲板、是左灯还是右灯，由 L3 按
 * 整车预测关联。
 */
struct Light {
  /// center 是 top 与 bottom 的中点，top 按图像 y 定，恒在 bottom 上方。
  cv::Point2f center{};

  /// 上端点
  cv::Point2f top{};

  /// 下端点
  cv::Point2f bottom{};

  /// 判出的颜色，见 lightColor
  ArmorColor color{ArmorColor::Unknown};

  /// 两端点的距离，单位为 pixel。
  double length{0.0};

  /// 端点连线偏离竖直方向的角度，单位为度。
  float tilt_angle_deg{0.0F};

  /// 在本帧灯条数组里的下标。
  std::size_t id{0};
};

/**
 * @brief L3 按整车预测投影出的一根灯条的上下端点，原图像素系
 *
 * 剖面搜索（searchLights）只在它附近找灯条，不做全区域二值化。L2 只拿它当
 * 搜索位置，不关心它是哪块板的哪根灯条——那仍由 L3 关联时决定。
 */
struct LightHint {
  /// 预测的上端点
  cv::Point2f top{};

  /// 预测的下端点
  cv::Point2f bottom{};
};

/**
 * @brief 一帧装甲感知的输出
 *
 * 网络检出的装甲板，以及沿预测灯条搜到、不属于任何检出装甲板的侧边灯条，
 * 后者交给 L3 做额外的端点观测。
 */
struct ArmorFrame {
  /// 网络检出的装甲板
  std::vector<Armor> armors;

  /// 侧边灯条
  std::vector<Light> lights;
};

/// 保留检测层原有接口名称，避免后端和测试因数据结构改名而失效。
using ArmorDetection = Armor;

}  // namespace L2Perception
