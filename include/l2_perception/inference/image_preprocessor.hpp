#pragma once

#include "l2_perception/inference/inference_backend.hpp"

#include <opencv2/core.hpp>

namespace L2Perception
{

// CPU 这里只负责保持宽高比的 letterbox；颜色、归一化和布局转换由 Backend 完成。
struct ImagePreprocessConfig
{
  cv::Scalar padding_color{114.0, 114.0, 114.0};
};

// 记录从原图到模型图的缩放与补边关系；Decoder 用它把模型关键点还原到原图。
// 当前实现使用“等比缩放 + 上下左右居中补边”，所以不能只保存一个 scale。
struct ImageTransform
{
  cv::Size source_size{};
  cv::Size model_size{};
  float source_to_model_scale{1.0F};
  int pad_left{0};
  int pad_top{0};
  int pad_right{0};
  int pad_bottom{0};

  // 原图点 -> 模型坐标：point * scale + 左上补边。
  [[nodiscard]] cv::Point2f sourceToModel(const cv::Point2f& point) const noexcept;
  // 模型点 -> 原图坐标：(point - 左上补边) / scale。
  [[nodiscard]] cv::Point2f modelToSource(const cv::Point2f& point) const noexcept;
};

struct PreprocessedImage
{
  InferenceInput input;
  ImageTransform transform;
};

class ImagePreprocessor
{
public:
  // 输入必须是相机输出的 BGR CV_8UC3；输出固定为连续 uint8 NHWC BGR。
  // 这里不调用推理 SDK，OpenVINO/TensorRT 可以复用完全相同的 letterbox 结果。
  [[nodiscard]] static PreprocessedImage run(
    const cv::Mat& image,
    const InferenceInputSpec& input_spec,
    const ImagePreprocessConfig& config = {});
};

}  // namespace L2Perception
