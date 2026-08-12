#pragma once

#include "l2_perception/armor.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"
#include "l2_perception/inference/inference_result.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace L2Perception
{

// 默认按当前 YOLOV5 模型的输出协议解码 armor.xml/yolov5.xml：
// 输出名 output、形状 [1, 25200, 22]。
// 每个候选字段为：8 个角点、1 个置信度 logit、4 个颜色分数、9 个车辆类别分数。
// 这些字段规则属于 ArmorDecoder，绝不能写进推理 Backend。
enum class ArmorTensorLayout
{
  CandidatesByFields,  // [1, candidate_count, field_count]
  FieldsByCandidates   // [1, field_count, candidate_count]
};

// NMS 排序使用哪个分数。当前模型用 sigmoid 后的 objectness。
enum class ArmorNmsScoreSource
{
  Objectness,
  ClassScore
};

struct ArmorDecoderConfig
{
  std::string output_name{"output"};
  ArmorTensorLayout tensor_layout{ArmorTensorLayout::CandidatesByFields};

  // 以下 offset 的单位都是“float 字段下标”，不是字节下标。
  // 当前模型布局：0~7 四角点，8 confidence，9~12 颜色，13~21 类别。
  std::size_t corner_offset{0};
  std::size_t confidence_index{8};
  std::size_t color_offset{9};
  std::size_t color_count{4};  // Blue、Red、Gray、Purple
  std::size_t class_offset{13};
  std::size_t class_count{9};  // G、1、2、3、4、5、O、Bs、Bb

  // 模型原始四点依次为左上、左下、右下、右上；Armor/PnP 则要求
  // 左上、右上、右下、左下。数组值是每个 Armor 目标点对应的模型点下标。
  std::array<std::size_t, 4> corner_order{0, 3, 2, 1};

  // 颜色通道约定：0=Blue、1=Red、2=Gray、3=Purple。
  int red_color_index{1};
  int blue_color_index{0};
  int class_id_offset{0};
  float confidence_threshold{0.7F};  // 进入 NMS 的门限。
  float minimum_confidence{0.8F};    // NMS 之后必须严格大于该值。
  float nms_iou_threshold{0.3F};
  ArmorNmsScoreSource nms_score_source{ArmorNmsScoreSource::Objectness};
  float nms_score_threshold{0.7F};
  bool confidence_is_logit{true};          // output[8] 是 objectness logit。
  bool coordinates_are_normalized{false};  // true 时角点 0~1，需先乘模型宽高。
  bool class_aware_nms{false};             // true 时不同颜色/类别候选不互相抑制。
};

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
