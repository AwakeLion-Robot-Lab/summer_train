#pragma once

#include "l2_perception/inference/inference_backend.hpp"

#include <opencv2/core.hpp>

namespace L2Perception
{

// CPU 这里只负责保持宽高比的 letterbox；颜色、归一化和布局转换由 Backend 完成。
enum class LetterboxAlignment
{
  TopLeft,
  Centered
};

struct ImagePreprocessConfig
{
  // SP-Vision YOLOV5 将缩放图贴在左上角，右侧/下侧补纯黑。
  cv::Scalar padding_color{0.0, 0.0, 0.0};
  LetterboxAlignment alignment{LetterboxAlignment::TopLeft};
};

// 记录从原图到模型图的缩放与补边关系；Decoder 用它把模型关键点还原到原图。
// SP 默认左上贴齐，但仍完整保存四边补边量，以支持显式选择 Centered。
struct ImageTransform
{
  cv::Size source_size{};
  cv::Size model_size{};
  // SP YOLOV5 将 resize scale 保持为 double，解码关键点时执行 float/double
  // 除法后再窄化为 Point2f。这里必须保留同一数值路径。
  double source_to_model_scale{1.0};
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
  [[nodiscard]] static PreprocessedImage run(const cv::Mat& image,
                                             const InferenceInputSpec& input_spec,
                                             const ImagePreprocessConfig& config = {});
};

}  // namespace L2Perception
