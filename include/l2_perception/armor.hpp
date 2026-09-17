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
// 与整板模型类别字段 argmax 的下标一致；Armor 里保留 class_id 方便记录，需要
// 语义时调用 armorClassFromId。
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
// 未知类别返回 nullopt，不猜板型。L3 的 armorTypeOf 由它派生，PnP 和端点
// 观测的板宽都按这一份映射取。
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

// 角点的来源。传统精修只在证据充分时才替换网络角点，下游和离线工具要能
// 区分两者。
enum class CornerSource {
  Network,  // 网络回归的原始角点
  Refined   // 已由 ROI 内的灯条端点替换
};

// 类别的来源。二次分类会改写 class_id，下游和离线工具要能区分两者。
enum class ClassSource {
  Network,  // 整板网络的类别 argmax
  Number    // 已由数字分类器重判
};

// 整板网络检出的装甲板，角点可能已被传统精修替换。
struct Armor {
  // 顺序固定为：左上、右上、右下、左下；PnP 必须沿用同一顺序。
  // 左上/左下是左灯条的上下端点，右上/右下是右灯条的，L3 按此拆成两根灯条。
  std::array<cv::Point2f, 4> corners{};
  // 精修前的网络原始角点，顺序同 corners。留着是为了能离线对比两条通路，
  // 否则无从验证精修是否真有收益。
  std::array<cv::Point2f, 4> network_corners{};
  CornerSource corner_source{CornerSource::Network};
  // 精修角点相对网络角点的最大位移，单位 pixel；未精修时为 0。
  float corner_shift{0.0F};
  // 网络角点的几何中心。精修不重算它，与 SP-Vision 同口径。
  cv::Point2f center{};
  // 车辆类别编号，见 ArmorClass。二次分类开着时它是数字分类器的结果。
  int class_id{-1};
  // 网络给出的原始类别，二次分类改写 class_id 之后还能离线对比两路。
  int network_class_id{-1};
  ClassSource class_source{ClassSource::Network};
  ArmorColor color{ArmorColor::Unknown};
  // 网络置信度（yolov5 为 sigmoid 后的 objectness）。
  float confidence{0.0F};
  // 数字分类器的 softmax 置信度；没跑二次分类时为 0。
  float number_confidence{0.0F};

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
  // 在本帧灯条数组里的下标。
  std::size_t id{0};
};

// 一帧装甲感知的输出：网络检出的装甲板，以及 light_roi 内不属于任何检出
// 装甲板的侧边灯条，后者交给 L3 做额外的端点观测。
struct ArmorFrame {
  std::vector<Armor> armors;
  std::vector<Light> lights;
};

// 保留检测层原有接口名称，避免后端和测试因数据结构改名而失效。
using ArmorDetection = Armor;

}  // namespace L2Perception
