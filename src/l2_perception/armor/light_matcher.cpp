#include "l2_perception/armor/light_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

bool colorMatches(const Light& first, const Light& second, ArmorColor color) noexcept
{
  if (color == ArmorColor::Unknown) {
    return first.color != ArmorColor::Unknown && first.color == second.color;
  }
  return first.color == color && second.color == color;
}

// 两灯条的四个端点围出轴对齐框，框里落进第三根灯条的任一端点或中心就返回
// true：中间隔着一根，说明这两根不在同一块板上。
bool containLight(std::size_t first, std::size_t second, const std::vector<Light>& lights)
{
  const cv::Rect bounding_rect = cv::boundingRect(std::vector<cv::Point2f>{
    lights[first].top, lights[first].bottom, lights[second].top, lights[second].bottom});
  for (std::size_t index = 0; index < lights.size(); ++index) {
    if (index == first || index == second) {
      continue;
    }
    const Light& test = lights[index];
    if (bounding_rect.contains(test.top) || bounding_rect.contains(test.bottom) ||
        bounding_rect.contains(test.center)) {
      return true;
    }
  }
  return false;
}

std::optional<LightPair> isArmor(
  std::size_t first, std::size_t second, const std::vector<Light>& lights,
  const LightMatcherConfig& config)
{
  const Light& light_1 = lights[first];
  const Light& light_2 = lights[second];
  // 长度比：两根灯条在同一块板上时长度接近，差太多多半配错了。
  const double longer = std::max(light_1.length, light_2.length);
  const double shorter = std::min(light_1.length, light_2.length);
  if (!(longer > 1.0) || shorter / longer <= config.min_light_length_ratio) {
    return std::nullopt;
  }

  // 中心距离按平均灯长归一化，落进小板或大板任一区间才算数。
  const cv::Point2f diff = light_1.center - light_2.center;
  const float center_distance = static_cast<float>(
    cv::norm(diff) / ((light_1.length + light_2.length) * 0.5));
  const bool small_ok = config.min_small_center_distance <= center_distance &&
                        center_distance < config.max_small_center_distance;
  const bool large_ok = config.min_large_center_distance <= center_distance &&
                        center_distance < config.max_large_center_distance;
  if (!small_ok && !large_ok) {
    return std::nullopt;
  }

  // 连线与水平方向的夹角。用 atan2 而不是 atan(dy / dx)，后者在两灯条竖直
  // 排列时会除零。
  const float angle_deg = static_cast<float>(
    std::atan2(std::abs(diff.y), std::abs(diff.x)) * 180.0 / CV_PI);
  if (!(angle_deg < config.max_pair_angle_deg)) {
    return std::nullopt;
  }

  const bool first_is_left = light_1.center.x < light_2.center.x;
  LightPair pair;
  pair.left = first_is_left ? first : second;
  pair.right = first_is_left ? second : first;
  pair.large = center_distance > config.min_large_center_distance;
  pair.center_distance = center_distance;
  return pair;
}

}  // namespace

std::vector<LightPair> matchLights(
  const std::vector<Light>& lights, ArmorColor color, const LightMatcherConfig& config)
{
  std::vector<LightPair> pairs;
  for (std::size_t first = 0; first < lights.size(); ++first) {
    for (std::size_t second = first + 1; second < lights.size(); ++second) {
      if (!colorMatches(lights[first], lights[second], color) ||
          containLight(first, second, lights)) {
        continue;
      }
      if (auto pair = isArmor(first, second, lights, config)) {
        pairs.push_back(*pair);
      }
    }
  }
  return pairs;
}

}  // namespace L2Perception
