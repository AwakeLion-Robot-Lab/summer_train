#pragma once

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core/types.hpp>

namespace L2Perception {

enum class ArmorColor {
  Red,
  Blue,
  Unknown
};

// 9 个车辆类别。数值是跨层约定的 class_id（G、1、2、3、4、5、O、Bs、Bb），
// 由数字分类器的标签映射而来；Armor 里保留 class_id 方便记录，需要语义时
// 调用 armorClassFromId。
enum class ArmorClass : int {
  Guard = 0,      // G，哨兵
  Hero = 1,       // 1，英雄
  Engineer = 2,   // 2，工程
  Infantry3 = 3,  // 3，步兵三号
  Infantry4 = 4,  // 4，步兵四号
  Infantry5 = 5,  // 5，步兵五号
  Outpost = 6,    // O，前哨站
  BaseSmall = 7,  // Bs，小装甲基地
  BaseLarge = 8,  // Bb，大装甲基地
  Unknown = -1
};

constexpr ArmorClass armorClassFromId(int class_id) noexcept
{
  return class_id >= static_cast<int>(ArmorClass::Guard)
             && class_id <= static_cast<int>(ArmorClass::BaseLarge)
           ? static_cast<ArmorClass>(class_id)
           : ArmorClass::Unknown;
}

// 识别类别 → 是否大装甲板。场上只有四板车，大装甲板仅英雄使用：平衡步兵已不存在，
// 基地虽然有 Bs/Bb 两个类别但装甲板实物都是小板。
//
// 未知类别返回 nullopt，不猜板型。L2 的灯条配对用它剔除「数字与两灯条间距
// 推出的板型矛盾」的候选，L3 的 armorTypeOf 也由它派生——两处必须同一份映射，
// 否则会出现 L2 当大板放行、L3 却按小板几何做 PnP 的情况。
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

// 由左右两根灯条配出、并经数字分类确认的装甲板。
struct Armor {
  // 顺序固定为：左上、右上、右下、左下；PnP 必须沿用同一顺序。
  // 四个角点就是左右灯条的上下端点。
  std::array<cv::Point2f, 4> corners{};
  // 四个角点在原图像素坐标系中的几何中心。
  cv::Point2f center{};
  // 车辆类别编号，见 ArmorClass。
  int class_id{-1};
  ArmorColor color{ArmorColor::Unknown};
  // 数字分类的 softmax 概率。
  float confidence{0.0F};

  Eigen::Vector3d xyz_in_barrel{Eigen::Vector3d::Zero()};  // 单位：m
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};   // 单位：m
  // 固定顺序为 [yaw, pitch, roll]。
  Eigen::Vector3d ypr_in_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};   // 方位角加距离
};

// 这根灯条由哪一路检出，见 LightMode。
enum class LightSource {
  Model,
  Classic
};

// 检出的单根灯条，坐标都在原图像素系。它不带车辆编号，只有端点观测要用的
// 上下端点和几何量；属于哪块装甲板、是左灯还是右灯，由 L3 按整车预测关联。
struct Light {
  // center 是 top 与 bottom 的中点，top 按图像 y 定，恒在 bottom 上方。
  cv::Point2f center{};
  cv::Point2f top{};
  cv::Point2f bottom{};
  ArmorColor color{ArmorColor::Unknown};
  // 两端点的距离，单位为 pixel。
  double length{0.0};
  // 端点连线偏离竖直方向的角度，单位为度。
  float tilt_angle_deg{0.0F};
  // 模型的类别分数；传统检出的没有分数，记 1。
  float score{0.0F};
  LightSource source{LightSource::Model};
  // 在本帧灯条数组里的下标，LightPair 用它指回来。
  std::size_t id{0};
};

// 一帧装甲感知的输出：通过数字分类的装甲板，以及交给 L3 做端点观测的灯条。
// 后者按 detectFrame 的 light_roi 筛过，不一定是配出装甲板的那些。
struct ArmorFrame {
  std::vector<Armor> armors;
  std::vector<Light> lights;
};

// 保留检测层原有接口名称，避免后端和测试因数据结构改名而失效。
using ArmorDetection = Armor;

}  // namespace L2Perception
