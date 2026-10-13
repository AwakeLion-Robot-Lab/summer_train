#pragma once

#include "l2_perception/armor.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"
#include "l2_perception/inference/inference_result.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace L2Perception
{

/**
 * @brief 整板模型输出张量的两种排布
 */
enum class ArmorTensorLayout
{
  CandidatesByFields,  ///< [1, candidate_count, field_count]
  FieldsByCandidates   ///< [1, field_count, candidate_count]
};

/**
 * @brief 一种模型的输出字段布局
 *
 * 整组由模型导出时定死，换模型必须整组换，所以它是一个整体而不是十几个各调
 * 各的开关。offset 的单位是 float 字段下标，不是字节。
 *
 * 默认值就是 SP-Vision（同济）yolov5.xml 的布局：输出名 output、[1, 25200, 22]，
 * 0~7 四角点，8 objectness logit，9~12 颜色，13~21 九类。
 */
struct ArmorTensorContract
{
  /// 输出节点名
  std::string output_name{"output"};

  /// 张量排布
  ArmorTensorLayout tensor_layout{ArmorTensorLayout::CandidatesByFields};

  /// 四角点 8 个字段的起始下标
  std::size_t corner_offset{0};

  /// 有值表示模型带独立的 objectness 字段；无值表示没有（YOLOv8 就是这样），
  /// 置信度改取类别分支的最大值。用 optional 而不是"枚举 + 下标"，让"没有
  /// objectness 却配了下标"这种矛盾状态写不出来。
  std::optional<std::size_t> objectness_index{8};

  /// 置信度字段是否为 logit，需要先过 sigmoid。YOLOv8 的分类头已经过了。
  bool confidence_is_logit{true};

  /// 颜色字段的起始下标
  std::size_t color_offset{9};

  /// 颜色字段个数：Blue、Red、Gray、Purple
  std::size_t color_count{4};

  /// 类别字段的起始下标
  std::size_t class_offset{13};

  /// 类别字段个数：G、1、2、3、4、5、O、Bs、Bb
  std::size_t class_count{9};

  /// 模型原始四点依次为左上、左下、右下、右上；Armor/PnP 要求左上、右上、
  /// 右下、左下。数组值是每个 Armor 目标点对应的模型点下标。
  std::array<std::size_t, 4> corner_order{0, 3, 2, 1};

  /// 红色在颜色字段里的下标。颜色通道约定：0=Blue、1=Red、2=Gray、3=Purple。
  int red_color_index{1};

  /// 蓝色在颜色字段里的下标
  int blue_color_index{0};

  /// 类别 argmax 下标加上它才是 ArmorClass 编号
  int class_id_offset{0};

  /// true 时角点 0~1，需先乘模型宽高。
  bool coordinates_are_normalized{false};

  /**
   * @brief 逐字段比较两个契约
   * @return 全部字段相等时返回 true
   */
  friend bool operator==(const ArmorTensorContract&, const ArmorTensorContract&) = default;

  /**
   * @brief 该契约要求模型至少提供多少个字段
   * @return 字段数
   */
  std::size_t fieldCount() const noexcept;
};

/**
 * @brief 与契约无关的筛选策略
 *
 * 这些可以按场地和距离调，改了不会让解码错位，只改变留下多少候选，所以只有
 * 这一组暴露给 YAML。
 */
struct ArmorDecoderConfig
{
  /// 输出字段布局
  ArmorTensorContract contract{};

  /// 进入 NMS 的门限（SP score_threshold_）
  float confidence_threshold{0.7F};

  /// NMS 后必须严格大于它（SP demo min_confidence）
  float minimum_confidence{0.8F};

  /// NMS 的 IoU 门限
  float nms_iou_threshold{0.3F};

  /// NMS 的分数门限
  float nms_score_threshold{0.7F};

  /// true 时不同颜色/类别的候选不互相抑制
  bool class_aware_nms{false};
};

/**
 * @brief yolov5_22 预设
 *
 * 已知的输出契约。换模型时选契约、调阈值，不要逐个字段手配 offset：下标由
 * 导出时定死，配错不报错，只会静默解出垃圾角点。
 *   yolov5_22  [1, 25200, 22]  输出名 output   同济 yolov5.xml / 深大 Infantry-v5n
 *   yolov8_21  [1, 21, 6300]   输出名 output0  深大 Infantry-v8n，无 objectness
 *
 * @return yolov5_22 的契约与默认阈值
 */
ArmorDecoderConfig yolov5Preset() noexcept;

/**
 * @brief yolov8_21 预设，见 yolov5Preset
 * @return yolov8_21 的契约与默认阈值
 */
ArmorDecoderConfig yolov8Preset() noexcept;

/**
 * @brief 按名字取预设
 * @param name 预设名，如 "yolov5_22"、"yolov8_21"
 * @return 对应的预设；名字无效时返回 nullopt，由调用方报错
 */
[[nodiscard]] std::optional<ArmorDecoderConfig> decoderPreset(std::string_view name);

/**
 * @brief 按模型的输出形状认预设
 *
 * 上游部署库 postprocess_mode=auto 的做法，给 inference.decoder.layout: auto
 * 和"模型由命令行临时指定"的离线工具用。认不出就抛，绝不退回默认值。
 *
 * @param outputs 模型的输出节点名和形状，见 probeOutputSpecs
 * @return 匹配的预设，output_name 换成模型实际的输出名
 */
[[nodiscard]] ArmorDecoderConfig decoderFor(const std::vector<InferenceOutputSpec>& outputs);

/**
 * @brief 整板模型输出解码器
 */
class ArmorDecoder
{
public:
  /**
   * @brief 用给定配置构造解码器
   * @param config 输出契约与筛选阈值
   */
  explicit ArmorDecoder(ArmorDecoderConfig config = {});

  /**
   * @brief 核对模型输出与契约是否一致
   *
   * 检查输出名存在、维度对、字段数恰好相等。不一致时抛异常，并在消息里点明
   * 最常见的原因（layout 与模型不配、或传成了灯条模型）。只在启动装配检测器
   * 时调用一次。
   *
   * @param outputs 模型的输出节点名和形状
   */
  void validate(const std::vector<InferenceOutputSpec>& outputs) const;

  /**
   * @brief 原始 float 输出 → 原图坐标的 Armor
   *
   * 依次做阈值、角点重排、颜色/类别 argmax、NMS。不做 PnP，也不排目标优先级。
   *
   * @param result 一次推理的输出
   * @param transform 预处理时的 letterbox 变换，用于把角点还原回原图
   * @return 检出的装甲板
   */
  [[nodiscard]] std::vector<Armor> decode(
    const InferenceResult& result, const ImageTransform& transform) const;

  /**
   * @brief 当前配置
   * @return 输出契约与筛选阈值
   */
  const ArmorDecoderConfig& config() const noexcept { return config_; }

private:
  /// 输出契约与筛选阈值
  ArmorDecoderConfig config_;
};

}  // namespace L2Perception
