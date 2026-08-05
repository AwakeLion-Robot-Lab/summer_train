#include "tools/camera_calibration/high_precision_calibrator.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace {

const char* kCommandLineKeys =
  "{help h usage ? |                     | 显示命令行帮助}"
  "{@input-dir     |                     | 相机采集生成的图像会话目录}"
  "{output-dir o   | calibration_results | 标定结果的根目录}"
  "{cols           | 11                  | 棋盘格水平方向的内角点数量}"
  "{rows           | 8                   | 棋盘格垂直方向的内角点数量}"
  "{square-size    | 15.0                 | 棋盘格单格边长，单位须保持一致}"
  "{max-views      | 1000                 | 最多使用的多样视角数量}"
  "{min-views      | 20                  | 标定所需的最少有效视角数量}"
  "{max-sharpness  | 5.0                 | 最大边缘过渡宽度，单位为像素，越小越清晰}"
  "{min-contrast   | 30.0                | 黑白区域的最小灰度差}"
  "{min-area       | 0.005               | 内角点凸包占图像面积的最小比例}"
  "{bootstrap      | 0                   | 参数重采样次数，0 表示关闭}"
  "{preview        | false               | 预览角点检测结果，按 q 或 Esc 终止}";

void printMatrix(const cv::Mat& matrix)
{
  std::cout << std::setprecision(12);
  for (int row = 0; row < matrix.rows; ++row) {
    std::cout << "  [";
    for (int column = 0; column < matrix.cols; ++column) {
      if (column > 0) {
        std::cout << ", ";
      }
      std::cout << matrix.at<double>(row, column);
    }
    std::cout << "]\n";
  }
}

int run(int argc, char* argv[])
{
  cv::CommandLineParser parser(argc, argv, kCommandLineKeys);
  parser.about(
    "使用相机采集图像进行高精度棋盘格相机标定。\n"
    "cols 和 rows 表示内角点数量，不是黑白方格数量。");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  tools::CameraCalibratorOptions options;
  options.input_directory = parser.get<std::string>(0);
  options.output_root = parser.get<std::string>("output-dir");
  options.pattern_size = {
    parser.get<int>("cols"), parser.get<int>("rows")};
  options.square_size = parser.get<double>("square-size");
  const int max_views = parser.get<int>("max-views");
  const int min_views = parser.get<int>("min-views");
  options.max_sharpness =
    parser.get<double>("max-sharpness");
  options.min_contrast = parser.get<double>("min-contrast");
  options.min_board_area_ratio = parser.get<double>("min-area");
  options.bootstrap_iterations = parser.get<int>("bootstrap");
  options.preview = parser.get<bool>("preview");
  if (!parser.check()) {
    parser.printErrors();
    return 2;
  }
  if (max_views <= 0 || min_views <= 0) {
    throw std::invalid_argument(
      "max-views and min-views must be positive");
  }
  options.max_views = static_cast<std::size_t>(max_views);
  options.min_views = static_cast<std::size_t>(min_views);
  options.progress = [](const std::string& message) {
    std::cout << "[camera_calibrator] " << message << std::endl;
  };

  const auto result = tools::calibrateCameraImages(options);
  std::cout << "\nCalibration finished\n"
            << "  input images: " << result.discovered_images << '\n'
            << "  checkerboards detected: " << result.detected_images << '\n'
            << "  quality passed: " << result.quality_images << '\n'
            << "  diversity selected: " << result.selected_images << '\n'
            << "  final views: " << result.used_images << '\n'
            << "  model: " << result.distortion_model << '\n'
            << "  RMS: " << result.rms << " px\n"
            << "  cross-validation RMS: "
            << result.cross_validation_rms << " px\n"
            << "  per-view p95: " << result.per_view_p95 << " px\n"
            << "  camera matrix:\n";
  printMatrix(result.camera_matrix);
  std::cout << "  distortion coefficients:\n";
  printMatrix(result.distortion_coefficients);
  for (const auto& warning : result.warnings) {
    std::cout << "  warning: " << warning << '\n';
  }
  std::cout << "  output: " << result.output_directory << '\n'
            << "  quality: "
            << (result.quality_passed ? "PASS" : "FAIL") << '\n';
  return result.quality_passed ? 0 : 3;
}

}  // namespace

int main(int argc, char* argv[])
{
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    cv::destroyAllWindows();
    std::cerr << "camera_calibrator failed: "
              << error.what() << '\n';
    return 1;
  }
}
