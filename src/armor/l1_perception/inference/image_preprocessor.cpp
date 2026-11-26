#include "l1_perception/inference/image_preprocessor.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace L2Perception
{
namespace
{

void validateInputSpec(const InferenceInputSpec& input_spec)
{
  // 公共输入严格固定为 U8 NHWC BGR；H/W 在下标 1/2，通道数在下标 3。
  if (input_spec.shape.size() != 4 || input_spec.shape[0] != 1 || input_spec.shape[1] == 0 ||
      input_spec.shape[2] == 0 || input_spec.shape[3] != 3) {
    throw std::invalid_argument(
        "ImagePreprocessor requires a static uint8 NHWC input shape [1, H, W, 3]");
  }
}

}  // namespace

cv::Point2f ImageTransform::sourceToModel(const cv::Point2f& point) const noexcept
{
  return {
    static_cast<float>(static_cast<double>(point.x) * source_to_model_scale + pad_left),
    static_cast<float>(static_cast<double>(point.y) * source_to_model_scale + pad_top)};
}

cv::Point2f ImageTransform::modelToSource(const cv::Point2f& point) const noexcept
{
  return {
    static_cast<float>((static_cast<double>(point.x) - pad_left) / source_to_model_scale),
    static_cast<float>((static_cast<double>(point.y) - pad_top) / source_to_model_scale)};
}

PreprocessedImage ImagePreprocessor::run(const cv::Mat& image, const InferenceInputSpec& input_spec,
                                         const ImagePreprocessConfig& config)
{
  PreprocessedImage output;
  runInto(image, input_spec, output, config);
  return output;
}

void ImagePreprocessor::runInto(const cv::Mat& image, const InferenceInputSpec& input_spec,
                                PreprocessedImage& output, const ImagePreprocessConfig& config)
{
  if (image.empty()) {
    throw std::invalid_argument("ImagePreprocessor received an empty image");
  }
  if (image.type() != CV_8UC3) {
    throw std::invalid_argument("ImagePreprocessor requires a BGR CV_8UC3 image");
  }

  validateInputSpec(input_spec);

  // input_spec.shape = {N, H, W, C}。
  const int model_height = static_cast<int>(input_spec.shape[1]);
  const int model_width = static_cast<int>(input_spec.shape[2]);
  // 取较小缩放比以完整保留原图比例，剩余区域由 padding 填充，即标准 letterbox。
  const double resize_scale =
      std::min(static_cast<double>(model_width) / static_cast<double>(image.cols),
               static_cast<double>(model_height) / static_cast<double>(image.rows));

  //浮点缩放结果直接截断为整数，不使用四舍五入。
  const int resized_width = std::max(1, static_cast<int>(image.cols * resize_scale));
  const int resized_height = std::max(1, static_cast<int>(image.rows * resize_scale));
  const int total_padding_x = model_width - resized_width;
  const int total_padding_y = model_height - resized_height;
  if (total_padding_x < 0 || total_padding_y < 0) {
    throw std::logic_error("ImagePreprocessor generated an invalid letterbox size");
  }

  // 内容放在 (0, 0)，padding 全部留在右/下。Centered 仅供其他
  // 显式配置的模型使用；无论哪种模式，偏移都会交给 Decoder 做逆变换。
  const int pad_left = config.alignment == LetterboxAlignment::Centered ? total_padding_x / 2 : 0;
  const int pad_right = total_padding_x - pad_left;
  const int pad_top = config.alignment == LetterboxAlignment::Centered ? total_padding_y / 2 : 0;
  const int pad_bottom = total_padding_y - pad_top;

  output.input.name = input_spec.name;
  output.input.shape = input_spec.shape;
  output.transform = {.source_size = image.size(),
                      .model_size = {model_width, model_height},
                      .source_to_model_scale = resize_scale,
                      .pad_left = pad_left,
                      .pad_top = pad_top,
                      .pad_right = pad_right,
                      .pad_bottom = pad_bottom};
  // 裸指针构造的 Mat 没有可延长的所有权，必须复制到独立存储。
  if (image.size() == cv::Size(model_width, model_height) && image.isContinuous() && image.u) {
    output.input.setExternalView(image.data, image.total() * image.elemSize(),
                                 std::make_shared<cv::Mat>(image));
    return;
  }

  // 固定模型尺寸时复用最终 U8 缓冲区，不清零内容区。
  const auto nhwc = output.input.resizeOwnedData(static_cast<std::size_t>(model_width) *
                                                static_cast<std::size_t>(model_height) * 3);
  cv::Mat letterboxed(model_height, model_width, CV_8UC3, nhwc.data());
  // 内容区会由 copyTo/resize 完整覆盖。只填当前四边，ROI、贴齐方式或颜色
  // 改变时也不会把上一帧内容留在新的补边区。
  if (pad_top > 0)
    letterboxed.rowRange(0, pad_top).setTo(config.padding_color);
  if (pad_bottom > 0)
    letterboxed.rowRange(model_height - pad_bottom, model_height).setTo(config.padding_color);
  if (pad_left > 0)
    letterboxed(cv::Rect(0, pad_top, pad_left, resized_height)).setTo(config.padding_color);
  if (pad_right > 0)
    letterboxed(cv::Rect(model_width - pad_right, pad_top, pad_right, resized_height))
      .setTo(config.padding_color);

  const cv::Rect content_rect(pad_left, pad_top, resized_width, resized_height);
  cv::Mat content = letterboxed(content_rect);
  if (image.size() == content.size()) {
    image.copyTo(content);
  } else {
    cv::resize(image, content, content.size(), 0.0, 0.0, cv::INTER_LINEAR);
  }
}

}  // namespace L2Perception
