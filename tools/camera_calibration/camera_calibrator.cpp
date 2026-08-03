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
  "{help h usage ? |                     | Show command line help}"
  "{@input-dir     |                     | Saver image session directory}"
  "{output-dir o   | calibration_results | Result directory root}"
  "{cols           | 11                  | Checkerboard inner-corner columns}"
  "{rows           | 8                   | Checkerboard inner-corner rows}"
  "{square-size    | 15.0                 | Checker square size in one consistent unit}"
  "{max-views      | 1000                 | Maximum diverse views used}"
  "{min-views      | 20                  | Minimum views required}"
  "{max-sharpness  | 3.0                 | Maximum edge transition width in pixels}"
  "{min-contrast   | 30.0                | Minimum black-white gray-level difference}"
  "{min-area       | 0.005               | Minimum inner-corner hull/image area ratio}"
  "{bootstrap      | 0                   | Parameter bootstrap iterations; 0 disables}"
  "{preview        | false               | Preview detected corners; q/Esc aborts}";

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
    "High-precision checkerboard camera calibration from saver images.\n"
    "Pattern dimensions are inner corners: an 11x8-square board is 10x7.");
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
