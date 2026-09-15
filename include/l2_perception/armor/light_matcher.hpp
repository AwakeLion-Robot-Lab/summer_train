#pragma once

#include "l2_perception/armor.hpp"

#include <cstddef>
#include <vector>

namespace L2Perception
{

// 灯条两两配对的几何门限，默认值取自 rm_auto_aim armor_detector 节点。
// 距离以两灯条平均长度为单位。
struct LightMatcherConfig
{
  // 短灯条 / 长灯条。
  float min_light_length_ratio{0.7F};
  float min_small_center_distance{0.8F};
  float max_small_center_distance{3.2F};
  float min_large_center_distance{3.2F};
  float max_large_center_distance{5.5F};
  // 两灯条中心连线偏离水平的最大角度，单位为度。
  float max_pair_angle_deg{35.0F};
};

// 一对可能构成装甲板的灯条。left/right 是灯条数组下标，已按图像 x 排好。
struct LightPair
{
  std::size_t left{0};
  std::size_t right{0};
  // 两灯条中心距离超过 min_large_center_distance 时按大装甲板抠数字。
  bool large{false};
  float center_distance{0.0F};
};

// 照搬 rm_auto_aim Detector::matchLights：只配对 color 颜色的灯条（Unknown 表示
// 两根颜色相同且已知即可），两灯条端点围成的框里夹着别的灯条时跳过，其余按
// 长度比、中心距离、连线角度筛。
//
// 这里只做几何判断，放宽到「可能是」为止：相邻两块板的灯条也会配成一对，
// 要靠后面的数字分类判成 negative 剔掉。
[[nodiscard]] std::vector<LightPair> matchLights(
  const std::vector<Light>& lights, ArmorColor color, const LightMatcherConfig& config);

}  // namespace L2Perception
