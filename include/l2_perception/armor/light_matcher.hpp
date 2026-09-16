#pragma once

#include "l2_perception/armor.hpp"

#include <cstddef>
#include <vector>

namespace L2Perception
{

// 灯条两两配对的几何门限。两个中心距离都以两灯条的平均长度为单位，所以门限
// 与目标远近无关。
struct LightMatcherConfig
{
  // 短灯条 / 长灯条，低于它说明两根不在同一块板上。
  float min_light_length_ratio{0.7F};
  float min_small_center_distance{0.8F};
  float max_small_center_distance{3.2F};
  float min_large_center_distance{3.2F};
  float max_large_center_distance{5.5F};
  // 两灯条中心连线偏离水平方向的最大角度，单位为度。
  float max_pair_angle_deg{35.0F};
};

// 一对可能构成装甲板的灯条。left/right 是传入数组的下标，已按图像 x 排好。
struct LightPair
{
  std::size_t left{0};
  std::size_t right{0};
  // 中心距离落在大板区间时为 true，数字分类按大板的宽高比抠图。
  bool large{false};
  float center_distance{0.0F};
};

// 枚举所有灯条两两组合，逐对判断：颜色（color 为 Unknown 时只要求两根同色
// 且已知）→ 两端点围成的框里不能夹着第三根灯条 → 长度比、中心距离、连线
// 角度三道几何门限。通过的按图像 x 定左右，并标出是大板还是小板。
//
// 判据只到「可能是一块板」为止：相邻两块板的灯条也会配成一对，由数字分类
// 判成 negative 剔除。
[[nodiscard]] std::vector<LightPair> matchLights(
  const std::vector<Light>& lights, ArmorColor color, const LightMatcherConfig& config);

}  // namespace L2Perception
