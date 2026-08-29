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

// 默认按 SP-Vision 的 YOLOV5 协议解码 armor.xml/yolov5.xml：
// 输出名 output、形状 [1, 25200, 22]。
// 每个候选字段为：8 个角点、1 个置信度 logit、4 个颜色分数、9 个车辆类别分数。
// 这些字段规则属于 ArmorDecoder，绝不能写进推理 Backend。
enum class ArmorTensorLayout
{
  CandidatesByFields,  // [1, candidate_count, field_count]
  FieldsByCandidates   // [1, field_count, candidate_count]
};

// 一种模型的输出字段布局。整组由模型导出时定死，换模型必须整组换，
// 所以它是一个整体而不是十几个可以各调各的独立开关。offset 的单位是
// “float 字段下标”，不是字节下标。
//
// SP YOLOV5 的布局为：0~7 四角点，8 objectness，9~12 颜色，13~21 类别。
struct ArmorTensorContract
{
  std::string output_name{"output"};
  ArmorTensorLayout tensor_layout{ArmorTensorLayout::CandidatesByFields};

  std::size_t corner_offset{0};
  // 有值表示模型带独立的 objectness 字段；无值表示没有（YOLOV8 就是这样），
  // 置信度改取类别分支的最大值。用 optional 而不是“枚举 + 下标”，是为了让
  // “没有 objectness 却仍配了下标”这种自相矛盾的状态根本无法被写出来。
  std::optional<std::size_t> objectness_index{8};
  // 置信度字段是否为 logit，需要先过 sigmoid。YOLOV8 的分类头已经过了。
  bool confidence_is_logit{true};

  std::size_t color_offset{9};
  std::size_t color_count{4};  // Blue、Red、Gray、Purple
  std::size_t class_offset{13};
  std::size_t class_count{9};  // G、1、2、3、4、5、O、Bs、Bb

  // 模型原始四点依次为左上、左下、右下、右上；Armor/PnP 则要求
  // 左上、右上、右下、左下。数组值是每个 Armor 目标点对应的模型点下标。
  std::array<std::size_t, 4> corner_order{0, 3, 2, 1};

  // SP-Vision YOLOV5 的颜色通道约定是 0=Blue、1=Red、2=Gray、3=Purple。
  int red_color_index{1};
  int blue_color_index{0};
  int class_id_offset{0};
  bool coordinates_are_normalized{false};  // true 时角点 0~1，需先乘模型宽高。

  // 契约是否匹配靠整体比较，不靠逐字段核对。
  [[nodiscard]] friend bool operator==(
    const ArmorTensorContract&, const ArmorTensorContract&) = default;

  // 该契约要求模型至少提供多少个字段。
  [[nodiscard]] std::size_t requiredFieldCount() const noexcept;
};

// 与契约无关的筛选策略。这些是可以按场地和距离自由调的数，改它们不会
// 让解码错位，只会改变留下多少候选——所以只有这一组暴露给 YAML。
struct ArmorDecoderConfig
{
  ArmorTensorContract contract{};

  float confidence_threshold{0.7F};  // SP score_threshold_：进入 NMS 的门限。
  float minimum_confidence{0.8F};    // SP demo min_confidence：NMS 后必须严格大于。
  float nms_iou_threshold{0.3F};
  float nms_score_threshold{0.7F};
  bool class_aware_nms{false};  // true 时不同颜色/类别候选不互相抑制。
};

// 已知的输出契约。换模型时选契约、调阈值，不要逐个字段手配 offset——下标由
// 模型导出时定死，配错不报错，只会静默解出垃圾角点。
//   yolov5_22  [1, 25200, 22]  输出名 output   SP yolov5.xml / 深大 Infantry-v5n
//   yolov8_21  [1, 21, 6300]   输出名 output0  深大 Infantry-v8n，无 objectness
[[nodiscard]] ArmorDecoderConfig yolov5_22DecoderConfig() noexcept;
[[nodiscard]] ArmorDecoderConfig yolov8_21DecoderConfig() noexcept;

// 按名字取预设，名字无效时返回 nullopt 由调用方报错。
[[nodiscard]] std::optional<ArmorDecoderConfig> armorDecoderPreset(std::string_view name);

// 按探测到的模型输出名取预设。给"模型在运行时才由命令行决定"的离线工具用；
// 实机 runtime 不猜，契约由 auto_aim.yaml 的 inference.decoder.layout 指定。
// 认不出就抛，绝不退回默认值——猜错契约只会静默解出垃圾角点。字段数对不对
// 由 ArmorDecoder::decode() 负责，这里不重复检查。
[[nodiscard]] ArmorDecoderConfig armorDecoderConfigFor(
  const std::vector<InferenceOutputSpec>& outputs);

class ArmorDecoder
{
public:
  explicit ArmorDecoder(ArmorDecoderConfig config = {});

  // 将模型坐标的原始 float 输出变为原图坐标 Armor。
  // 这里完成阈值、固定角点重排、颜色/类别 argmax、NMS；不会做 PnP 或目标优先级。
  [[nodiscard]] std::vector<Armor> decode(const InferenceResult& result,
                                          const ImageTransform& transform) const;

private:
  ArmorDecoderConfig config_;
};

}  // namespace L2Perception
