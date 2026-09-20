#pragma once

#include "l2_perception/armor.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 传统灯条检测的门限，判定顺序见 findLights。
struct LightFinderConfig
{
  // 灰度二值化阈值，低曝光画面取 100~140。过曝画面背景（海报之类）也会过阈，
  // 但多出来的轮廓在后面的长宽比和倾角筛选里就被挡住了，八段录像全程 140 都
  // 能出灯条。换曝光或场地亮度后仍要重新扫。
  int binary_threshold{140};
  // 最小外接矩形的 短边 / 长边，落在区间外的斑点不是灯条。
  float min_ratio{0.08F};
  float max_ratio{0.4F};
  // 端点连线偏离竖直方向的最大角度，单位为度。
  float max_angle_deg{40.0F};
  // 灯条长度下限，单位为 pixel。
  float min_length{4.0F};
  // 剔除属于已检出装甲板的灯条时，板外接框四周外扩多少倍灯条长度，见 insideArmor。
  float armor_margin{0.5F};
  // 见 lightColor：R/B 能量比大于它判红，小于它的倒数判蓝。
  double color_ratio_threshold{1.10};

  // 二值化底图换成颜色差分（蓝 B−R、红 R−B），与 ArmorRefiner 同一套做法和
  // 同一个理由：过曝灯条的核心 R≈G≈B，灰度阈值切到的是饱和白核，白核大小
  // 由曝光决定；差分图里白核归零，切到的是接近真实发光边界的彩色边缘。
  // lightColor 的注释早就写着"颜色只留在边缘光晕里"，这里只是把二值化也
  // 挪到同一个依据上。
  //
  // 要目标颜色才有减的方向，findLights 的 color 为 Unknown 时退回灰度。
  bool color_channel_diff{true};
  // color_channel_diff 生效时的阈值，语义与 binary_threshold 完全不同。
  int color_diff_threshold{50};
};

// 判定一根灯条的颜色：在端点连线周围取一块比灯条略宽的框，累加非饱和、非过暗
// 像素的 R 与 B 通道，按 red / blue 与 ratio_threshold 比较。
//
// 跳过饱和像素是必需的：过曝的灯条核心是白的（R≈G≈B），颜色只留在边缘光晕里。
[[nodiscard]] ArmorColor lightColor(
  const cv::Mat& image, const cv::Point2f& top, const cv::Point2f& bottom,
  double ratio_threshold);

// 传统灯条检测：在 roi 内二值化 → 取外轮廓 → 最小外接矩形 → 按 y 排序角点
// 取上下短边中点作端点 → 按长度、长宽比、倾角筛选 → lightColor 判颜色。
// 返回的坐标在原图上。
//
// color 是要找的目标颜色，只用来决定差分方向；Unknown 时退回灰度二值化，
// 所以缺省参数下行为与引入 color_channel_diff 之前完全一致。筛完之后仍由
// lightColor 独立判色，这里不拿 color 去认定结果。
[[nodiscard]] std::vector<Light> findLights(
  const cv::Mat& image, const cv::Rect& roi, const LightFinderConfig& config,
  ArmorColor color = ArmorColor::Unknown);

}  // namespace L2Perception
