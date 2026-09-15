#pragma once

#include "l2_perception/armor.hpp"
#include "l2_perception/inference/image_preprocessor.hpp"
#include "l2_perception/inference/inference_result.hpp"

#include <vector>

#include <opencv2/core.hpp>

namespace L2Perception
{

// 灯条关键点模型的输出契约。模型是 Ultralytics YOLOv8n-pose，1 类 light_bar，
// kpt_shape [2, 3]：
//   输出 output0 [1, 11, A]，channels-first
//   0~3   box (cx, cy, w, h)，模型输入像素坐标，不是归一化
//   4     类别分数，导出时已过 sigmoid
//   5~10  两个关键点 (x, y, v)，同样是模型输入像素坐标
struct LightDecoderConfig
{
  float score_threshold{0.25F};
  float nms_iou_threshold{0.45F};
  // 灯条周围 R/B 能量比大于它判红、小于它的倒数判蓝，中间判未知。
  double color_ratio_threshold{1.10};
};

class LightDecoder
{
public:
  explicit LightDecoder(LightDecoderConfig config = {});

  // 启动阶段校验模型输出形状。契约不符时抛出带原因的异常：配错模型不会让
  // 解码失败，只会解出垃圾端点，所以必须在加载时就挡住。
  static void validateOutputs(const std::vector<InferenceOutputSpec>& outputs);

  // source 是送进网络的那张 BGR 图（整图或 ROI 裁剪）；返回的灯条坐标也在
  // 这张图上，调用方负责补 ROI 偏移。
  [[nodiscard]] std::vector<Light> decode(
    const InferenceResult& result, const ImageTransform& transform,
    const cv::Mat& source) const;

  const LightDecoderConfig& config() const noexcept { return config_; }

private:
  LightDecoderConfig config_;
};

}  // namespace L2Perception
