#pragma once

#include "l2_perception/armor.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 侧边灯条只用剖面搜索找：沿 L3 预测的每根灯条做垂直剖面，找亮度脊线再定
// 端点（searchLights）。不二值化、不看轮廓形状，过曝只剩两道彩边、斜看变细、
// 与主灯条粘连都不受影响；代价是依赖预测，预测偏出搜索窗就找不到。原先的
// 整块二值化找轮廓（findLights）在回放上几乎采纳不到灯条，已删除。

// 剖面搜索的门限，判定顺序见 searchLights。
struct LightFinderConfig
{
  // 拟合直线偏离预测灯条方向的最大角度，单位为度。参考是预测方向而不是图像
  // 竖直方向，灯条在图像里斜多少都不受限。
  float max_angle_deg{40.0F};
  // 灯条长度下限，单位为 pixel。
  float min_length{4.0F};
  // 剔除属于已检出装甲板的灯条时，板外接框四周外扩多少倍灯条长度，见 insideArmor。
  float armor_margin{0.5F};
  // 见 lightColor：R/B 能量比大于它判红，小于它的倒数判蓝。
  double color_ratio_threshold{1.10};

  // 垂直灯条方向的搜索半宽：max(下限, 系数 × 预测灯条长度)，单位 pixel。要盖住
  // 预测的横向误差，又不能宽到把相邻灯条（3 m 处约 4 倍灯长）圈进来。
  float profile_half_width_ratio{0.5F};
  float profile_half_width_min_px{4.0F};
  // 沿灯条方向在预测端点外各多搜多少，同样按预测长度给，下限 2 px。端点落在
  // 搜索范围边上说明灯条没收住（可能连着别的亮斑），整根不要。
  float profile_extend_ratio{0.5F};
  // 一条剖面上 峰值 − 剖面最小值 至少多少才算这一行有灯条，0~255 灰度。
  float profile_min_contrast{40.0F};
  // 目标颜色通道与 G 通道都不低于它的像素当作过曝白核（底图取 255），见
  // searchLights。
  float profile_saturation{245.0F};
  // 非饱和像素的颜色差分乘上它，让白核比光晕高一档，见 searchLights。
  float profile_diff_gain{0.5F};
  // 端点定在行峰值降到这段亮行中位对比度的多少倍处。0.5 即半高；光晕沿灯条
  // 方向也会铺开，半高处的端点偏外。
  float profile_end_level{0.5F};
  // 各行脊线中心对直线拟合的 RMS 残差上限，单位 pixel。脊线弯折或跳到另一根
  // 灯条上时残差会突增。
  float profile_max_residual_px{1.0F};
};

// 剖面搜索：对每个 hint，沿预测灯条方向逐像素取垂直剖面，在剖面上找亮峰，
// 取峰上半高以上部分的质心作该行脊线中心（过曝的平顶也有定义，抛物线拟合
// 在平顶上会失效）；最长的一段连续亮行拟合成直线，端点按行峰值降到这段中位
// 数的 profile_end_level 倍处线性插值到亚像素。
//
// 底图是"饱和感知的颜色差分"：目标通道与 G 都饱和的像素（过曝白核）取 255，
// 其余取 profile_diff_gain ×（目标 − 对方）（蓝灯 B−R）。单用目标通道分不开
// 白色背景，单用差分会把白核挖空。color 为 Unknown 时取灰度。颜色由 lightColor
// 在光晕上另判。
//
// 每条剖面先 [1 2 1] 平滑，半高以上若有好几段，取离预测轴线最近的一段（而不
// 是最亮的一段）；直线拟合残差超限时逐个剔掉最差的行，剔到不足六成就放弃。
//
// 两个 hint 找到同一根灯条时只留对比度高的那根，免得 L3 把一根灯条配给两个
// 槽位。返回的坐标在原图上，top 恒在 bottom 上方。
// 一个 hint 的剖面搜索带在原图上的四个角：沿预测灯条向两端各外扩
// max(2, profile_extend_ratio × 灯长)，横向 ±max(profile_half_width_min_px,
// profile_half_width_ratio × 灯长) 取整。与 searchLights 用的是同一套尺寸，给
// 调试显示画"到底在哪找"。灯长不足 1 px 时 hint 无效，返回空。
[[nodiscard]] std::vector<cv::Point2f> searchBand(
  const LightHint& hint, const LightFinderConfig& config);

[[nodiscard]] std::vector<Light> searchLights(
  const cv::Mat& image, const std::vector<LightHint>& hints, const LightFinderConfig& config,
  ArmorColor color = ArmorColor::Unknown);

// 判定一根灯条的颜色：在端点连线周围取一块比灯条略宽的框，累加非饱和、非过暗
// 像素的 R 与 B 通道，按 red / blue 与 ratio_threshold 比较。
//
// 跳过饱和像素是必需的：过曝的灯条核心是白的（R≈G≈B），颜色只留在边缘光晕里。
[[nodiscard]] ArmorColor lightColor(
  const cv::Mat& image, const cv::Point2f& top, const cv::Point2f& bottom,
  double ratio_threshold);

}  // namespace L2Perception
