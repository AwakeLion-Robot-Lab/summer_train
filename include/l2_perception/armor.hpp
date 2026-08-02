#pragma once

#include <Eigen/Core>

#include <array>

#include <opencv2/core/types.hpp>

namespace L2Perception {

enum class ArmorColor {
  Red,
  Blue,
  Unknown
};

// Fosu armor.xml 的 9 个车辆类别，数值与模型第 13~21 字段 argmax 的下标一致。
// class_id 仍保留在 Armor 中，方便记录原始模型编号；
// 需要语义时调用此函数。
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

[[nodiscard]] constexpr ArmorClass armorClassFromId(int class_id) noexcept
{
  return class_id >= static_cast<int>(ArmorClass::Guard)
             && class_id <= static_cast<int>(ArmorClass::BaseLarge)
           ? static_cast<ArmorClass>(class_id)
           : ArmorClass::Unknown;
}

struct Armor {
  // 顺序固定为：左上、右上、右下、左下；PnP 必须沿用同一顺序。
  std::array<cv::Point2f, 4> corners{};
  // Fosu 约定：0~8 分别为 G、1、2、3、4、5、O、Bs、Bb。
  int class_id{-1};
  ArmorColor color{ArmorColor::Unknown};
  float confidence{0.0F};

  Eigen::Vector3d xyz_in_barrel{Eigen::Vector3d::Zero()};  // 单位：m
  Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};   // 单位：m
  Eigen::Vector3d rpy_in_barrel{Eigen::Vector3d::Zero()};  // [roll,pitch,yaw]，rad
  Eigen::Vector3d rpy_in_world{Eigen::Vector3d::Zero()};   // [roll,pitch,yaw]，rad
  Eigen::Vector3d ypd_in_world{Eigen::Vector3d::Zero()};   // 球坐标系
};

// 保留检测层原有接口名称，避免后端和测试因数据结构改名而失效。
using ArmorDetection = Armor;

}  // namespace L2Perception
