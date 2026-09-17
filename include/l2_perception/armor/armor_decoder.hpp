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

// 整板模型输出张量的两种排布。
enum class ArmorTensorLayout
{
  CandidatesByFields,  // [1, candidate_count, field_count]
  FieldsByCandidates   // [1, field_count, candidate_count]
};

// 一种模型的输出字段布局。整组由模型导出时定死，换模型必须整组换，所以它是
// 一个整体而不是十几个各调各的开关。offset 的单位是 float 字段下标，不是字节。
//
// 默认值就是 SP-Vision（同济）yolov5.xml 的布局：输出名 output、[1, 25200, 22]，
// 0~7 四角点，8 objectness logit，9~12 颜色，13~21 九类。
struct ArmorTensorContract
{
  std::string output_name{"output"};
  ArmorTensorLayout tensor_layout{ArmorTensorLayout::CandidatesByFields};

  std::size_t corner_offset{0};
  // 有值表示模型带独立的 objectness 字段；无值表示没有（YOLOv8 就是这样），
  // 置信度改取类别分支的最大值。用 optional 而不是“枚举 + 下标”，让“没有
  // objectness 却配了下标”这种矛盾状态写不出来。
  std::optional<std::size_t> objectness_index{8};
  // 置信度字段是否为 logit，需要先过 sigmoid。YOLOv8 的分类头已经过了。
  bool confidence_is_logit{true};

  std::size_t color_offset{9};
  std::size_t color_count{4};  // Blue、Red、Gray、Purple
  std::size_t class_offset{13};
  std::size_t class_count{9};  // G、1、2、3、4、5、O、Bs、Bb

  // 模型原始四点依次为左上、左下、右下、右上；Armor/PnP 要求左上、右上、
  // 右下、左下。数组值是每个 Armor 目标点对应的模型点下标。
  std::array<std::size_t, 4> corner_order{0, 3, 2, 1};

  // 颜色通道约定：0=Blue、1=Red、2=Gray、3=Purple。
  int red_color_index{1};
  int blue_color_index{0};
  int class_id_offset{0};
  bool coordinates_are_normalized{false};  // true 时角点 0~1，需先乘模型宽高。

  friend bool operator==(const ArmorTensorContract&, const ArmorTensorContract&) = default;

  // 该契约要求模型至少提供多少个字段。
  std::size_t fieldCount() const noexcept;
};

// 与契约无关的筛选策略。这些可以按场地和距离调，改了不会让解码错位，只改变
// 留下多少候选，所以只有这一组暴露给 YAML。
struct ArmorDecoderConfig
{
  ArmorTensorContract contract{};

  float confidence_threshold{0.7F};  // 进入 NMS 的门限（SP score_threshold_）
  float minimum_confidence{0.8F};    // NMS 后必须严格大于它（SP demo min_confidence）
  float nms_iou_threshold{0.3F};
  float nms_score_threshold{0.7F};
  bool class_aware_nms{false};  // true 时不同颜色/类别的候选不互相抑制
};

// 已知的输出契约。换模型时选契约、调阈值，不要逐个字段手配 offset：下标由
// 导出时定死，配错不报错，只会静默解出垃圾角点。
//   yolov5_22  [1, 25200, 22]  输出名 output   同济 yolov5.xml / 深大 Infantry-v5n
//   yolov8_21  [1, 21, 6300]   输出名 output0  深大 Infantry-v8n，无 objectness
ArmorDecoderConfig yolov5Preset() noexcept;
ArmorDecoderConfig yolov8Preset() noexcept;

// 按名字取预设，名字无效时返回 nullopt 由调用方报错。
[[nodiscard]] std::optional<ArmorDecoderConfig> decoderPreset(std::string_view name);

// 按模型的输出名认预设，给“模型由命令行临时指定”的离线工具用；实机 runtime
// 不猜，契约由 auto_aim.yaml 的 inference.decoder.layout 指定。认不出就抛，
// 绝不退回默认值。
[[nodiscard]] ArmorDecoderConfig decoderFor(const std::vector<InferenceOutputSpec>& outputs);

class ArmorDecoder
{
public:
  explicit ArmorDecoder(ArmorDecoderConfig config = {});

  // 核对模型输出与契约是否一致：输出名存在、维度和字段数够用。不一致时抛异常，
  // 并在消息里点明最常见的原因（layout 与模型不配、或传成了灯条模型）。只在
  // 启动装配检测器时调用一次。
  void validate(const std::vector<InferenceOutputSpec>& outputs) const;

  // 原始 float 输出 → 原图坐标的 Armor：阈值、角点重排、颜色/类别 argmax、NMS。
  // 不做 PnP，也不排目标优先级。
  [[nodiscard]] std::vector<Armor> decode(
    const InferenceResult& result, const ImageTransform& transform) const;

  const ArmorDecoderConfig& config() const noexcept { return config_; }

private:
  ArmorDecoderConfig config_;
};

}  // namespace L2Perception
