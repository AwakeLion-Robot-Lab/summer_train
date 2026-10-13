#pragma once

#include "l2_perception/inference/inference_backend.hpp"

#include <opencv2/core.hpp>

namespace L2Perception
{

/**
 * @brief letterbox 时缩放后的图贴在哪
 *
 * CPU 这里只负责保持宽高比的 letterbox；颜色、归一化和布局转换由 Backend 完成。
 */
enum class LetterboxAlignment
{
  TopLeft,
  Centered
};

/**
 * @brief letterbox 预处理参数
 */
struct ImagePreprocessConfig
{
  /// 补边颜色。默认把缩放后的图贴在左上角，右侧和下侧补纯黑。
  cv::Scalar padding_color{0.0, 0.0, 0.0};

  /// 贴齐方式
  LetterboxAlignment alignment{LetterboxAlignment::TopLeft};
};

/**
 * @brief 从原图到模型图的缩放与补边关系
 *
 * 解码时用它把模型输出的关键点还原回原图。四边补边量都存下来，居中贴齐时
 * 也适用。
 */
struct ImageTransform
{
  /// 原图尺寸
  cv::Size source_size{};

  /// 模型输入尺寸
  cv::Size model_size{};

  /// 缩放系数保持 double：解码关键点时先做 float/double 除法再窄化成
  /// Point2f，换成 float 会改变数值结果。
  double source_to_model_scale{1.0};

  /// 左侧补边，pixel
  int pad_left{0};

  /// 上侧补边，pixel
  int pad_top{0};

  /// 右侧补边，pixel
  int pad_right{0};

  /// 下侧补边，pixel
  int pad_bottom{0};

  /**
   * @brief 原图点 -> 模型坐标：point * scale + 左上补边
   * @param point 原图像素坐标
   * @return 模型输入上的像素坐标
   */
  cv::Point2f sourceToModel(const cv::Point2f& point) const noexcept;

  /**
   * @brief 模型点 -> 原图坐标：(point - 左上补边) / scale
   * @param point 模型输入上的像素坐标
   * @return 原图像素坐标
   */
  cv::Point2f modelToSource(const cv::Point2f& point) const noexcept;
};

/**
 * @brief 一帧预处理的结果
 */
struct PreprocessedImage
{
  /// 交给后端的输入
  InferenceInput input;

  /// 原图到模型图的变换，解码时用
  ImageTransform transform;
};

/**
 * @brief 保持宽高比的 letterbox 预处理
 */
class ImagePreprocessor
{
public:
  /**
   * @brief 把一帧图像 letterbox 到模型输入尺寸
   *
   * 这里不调用推理 SDK，OpenVINO/TensorRT 可以复用完全相同的 letterbox 结果。
   *
   * @param image 必须是相机输出的 BGR CV_8UC3
   * @param input_spec 模型输入规格，取其中的 NHWC 宽高
   * @param config 补边颜色和贴齐方式
   * @return 连续 uint8 NHWC BGR 输入，以及原图到模型图的变换
   */
  [[nodiscard]] static PreprocessedImage run(const cv::Mat& image,
                                             const InferenceInputSpec& input_spec,
                                             const ImagePreprocessConfig& config = {});
};

}  // namespace L2Perception
