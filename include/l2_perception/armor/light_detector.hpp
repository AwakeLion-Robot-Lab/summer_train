#pragma once

#include "l2_perception/armor.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"
#include "l2_perception/inference/inference_result.hpp"

#include <optional>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 侧边灯条由哪一路给出。
//   Model   只跑关键点模型；
//   Classic 只跑传统二值化；
//   Hybrid  两路都跑，由 mergeLights 合并。
enum class LightMode {
  Model,
  Classic,
  Hybrid
};

// LightMode 与配置里的名字（model / classic / hybrid）互转，名字无效时返回 nullopt。
[[nodiscard]] std::string_view lightModeName(LightMode mode) noexcept;
[[nodiscard]] std::optional<LightMode> parseLightMode(std::string_view name) noexcept;

// 关键点模型的输出契约：Ultralytics YOLOv8n-pose，1 类 light_bar，kpt_shape [2, 3]，
// 输出 output0 形状 [1, 11, A]，channels-first：
//   0~3   box (cx, cy, w, h)，单位是模型输入像素，不是归一化坐标
//   4     类别分数，导出时已过 sigmoid
//   5~10  两个关键点 (x, y, v)，同样是模型输入像素
struct LightDecoderConfig
{
  // 低于 score_threshold 的 anchor 不进候选，其余按 nms_iou_threshold 做 NMS。
  float score_threshold{0.25F};
  float nms_iou_threshold{0.45F};
  // 见 lightColor：R/B 能量比大于它判红，小于它的倒数判蓝。
  double color_ratio_threshold{1.10};
};

// 传统灯条检测的门限，判定顺序见 findLights。
struct LightFinderConfig
{
  LightMode mode{LightMode::Hybrid};
  // 灰度二值化阈值。低曝光画面取 100~140；过曝画面背景也会过阈，要调到 240
  // 附近，并把 max_ratio 一起放宽到 0.8。
  int binary_threshold{140};
  // 最小外接矩形的 短边 / 长边，落在区间外的斑点不是灯条。
  float min_ratio{0.08F};
  float max_ratio{0.4F};
  // 端点连线偏离竖直方向的最大角度，单位为度。
  float max_angle_deg{40.0F};
  // 灯条长度下限，单位为 pixel。
  float min_length{4.0F};
  // mergeLights 的判重半径，单位是两根灯条中较长者的长度。同一块板的两根灯条
  // 中心至少相距约 2 倍灯长，取 0.5 不会把一对灯条并掉。
  float merge_radius{0.5F};
  // mergeLights 判重之后的长度一致性门限，见该函数。
  float length_agree{0.8F};
  // 剔除属于已检出装甲板的灯条时，板外接框四周外扩多少倍灯条长度，见 insideArmor。
  float armor_margin{0.5F};

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

// 关键点模型输出 → 灯条。
class LightDecoder
{
public:
  explicit LightDecoder(LightDecoderConfig config = {});

  // 核对模型输出是否符合 LightDecoderConfig 注释里的形状契约，不符时抛异常。
  // 只在启动装配检测器时调用一次。
  static void validate(const std::vector<InferenceOutputSpec>& outputs);

  // 解码一次推理结果：按分数筛 anchor → NMS → 关键点经 transform 映射回
  // source 的坐标系 → 按图像 y 定上下端点 → 算长度、倾角并判颜色。
  //
  // source 是送进网络的那张 BGR 图（整图或 ROI 裁剪），返回的坐标在这张图上，
  // 补 ROI 偏移是调用方的事。
  [[nodiscard]] std::vector<Light> decode(
    const InferenceResult& result, const ImageTransform& transform,
    const cv::Mat& source) const;

  const LightDecoderConfig& config() const noexcept { return config_; }

private:
  LightDecoderConfig config_;
};

// 传统灯条检测：在 roi 内二值化 → 取外轮廓 → 最小外接矩形 → 按 y 排序角点
// 取上下短边中点作端点 → 按长度、长宽比、倾角筛选 → lightColor 判颜色。
// 返回的坐标在原图上，source 字段为 Classic。
//
// color 是要找的目标颜色，只用来决定差分方向；Unknown 时退回灰度二值化，
// 所以缺省参数下行为与引入 color_channel_diff 之前完全一致。筛完之后仍由
// lightColor 独立判色，这里不拿 color 去认定结果。
[[nodiscard]] std::vector<Light> findLights(
  const cv::Mat& image, const cv::Rect& roi, const LightFinderConfig& config,
  double color_ratio_threshold, ArmorColor color = ArmorColor::Unknown);

// 合并两路灯条，传统的在前：每根模型灯条找 merge_radius 内最近的传统灯条，
// 找不到就追加；找到但两者长度的 短/长 低于 length_agree（二值化把灯条断开
// 或和背景粘连了）就替换掉那根传统灯条，否则丢弃。
[[nodiscard]] std::vector<Light> mergeLights(
  std::vector<Light> classic, const std::vector<Light>& model, float merge_radius,
  float length_agree);

}  // namespace L2Perception
