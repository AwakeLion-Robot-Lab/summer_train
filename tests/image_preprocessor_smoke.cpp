#include "l2_perception/inference/image_preprocessor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

[[nodiscard]] bool near(float actual, float expected, float tolerance = 1e-5F)
{
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] std::size_t pixelOffset(int x, int y, int width)
{
  return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
          static_cast<std::size_t>(x)) *
         3;
}

}  // namespace

int main()
{
  try {
    // 4x3 -> 8x6，内容贴在左上角，底部两行补黑。
    const cv::Mat source(3, 4, CV_8UC3, cv::Scalar{10, 20, 30});
    const L2Perception::InferenceInputSpec input_spec{.name = "images", .shape = {1, 8, 8, 3}};
    const auto output = L2Perception::ImagePreprocessor::run(source, input_spec);

    require(output.input.isConsistent(), "preprocessed tensor is inconsistent");
    require(output.transform.source_to_model_scale == 2.0F, "resize scale is wrong");
    require(output.transform.pad_left == 0 && output.transform.pad_top == 0,
            "letterbox must be anchored at the top-left corner");
    require(output.transform.pad_right == 0 && output.transform.pad_bottom == 2,
            "letterbox padding dimensions are wrong");

    const auto values = output.input.values();
    const std::size_t content_pixel = pixelOffset(7, 5, 8);
    require(values[content_pixel] == 10 && values[content_pixel + 1] == 20 &&
                values[content_pixel + 2] == 30,
            "resized content did not remain BGR on the host side");
    const std::size_t padding_pixel = pixelOffset(0, 6, 8);
    require(values[padding_pixel] == 0 && values[padding_pixel + 1] == 0 &&
                values[padding_pixel + 2] == 0,
            "letterbox padding must be black");

    const cv::Point2f source_point{1.5F, 1.0F};
    const cv::Point2f model_point = output.transform.sourceToModel(source_point);
    const cv::Point2f restored_point = output.transform.modelToSource(model_point);
    require(near(model_point.x, 3.0F) && near(model_point.y, 2.0F),
            "top-left letterbox point transform is wrong");
    require(near(restored_point.x, source_point.x) && near(restored_point.y, source_point.y),
            "letterbox inverse transform is wrong");

    // 验证缩放使用截断而不是四舍五入：7x5 -> 10x7，底部剩余 3 行。
    const cv::Mat uneven_source(5, 7, CV_8UC3, cv::Scalar{1, 2, 3});
    const L2Perception::InferenceInputSpec uneven_spec{.name = "images", .shape = {1, 10, 10, 3}};
    const auto uneven = L2Perception::ImagePreprocessor::run(uneven_source, uneven_spec);
    require(uneven.transform.pad_right == 0 && uneven.transform.pad_bottom == 3,
            "resize dimensions must use integer truncation");

    // 非默认 Centered 模式仍保存精确偏移，确保通用接口没有退化。
    L2Perception::ImagePreprocessConfig centered_config;
    centered_config.padding_color = {114, 114, 114};
    centered_config.alignment = L2Perception::LetterboxAlignment::Centered;
    const auto centered = L2Perception::ImagePreprocessor::run(source, input_spec, centered_config);
    require(centered.transform.pad_top == 1 && centered.transform.pad_bottom == 1,
            "explicit centered letterbox is wrong");
    const std::size_t centered_padding = pixelOffset(0, 0, 8);
    require(centered.input.values()[centered_padding] == 114,
            "explicit centered padding color is wrong");

    std::cout << "image preprocessor smoke passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "image preprocessor smoke failed: " << error.what() << '\n';
    return 1;
  }
}
