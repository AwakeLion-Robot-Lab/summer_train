#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace tools {

struct CameraCalibratorOptions {
  std::filesystem::path input_directory;
  std::filesystem::path output_root{"calibration_results"};
  cv::Size pattern_size{10, 7};
  double square_size{15.0};
  std::size_t max_views{1000};
  std::size_t min_views{20};
  double max_sharpness{3.0};
  double min_contrast{50.0};
  double min_board_area_ratio{0.005};
  int bootstrap_iterations{20};
  bool preview{false};
  std::function<void(const std::string&)> progress;
};

struct CameraCalibratorResult {
  std::filesystem::path output_directory;
  cv::Size image_size{};
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  std::string distortion_model;
  std::size_t discovered_images{};
  std::size_t detected_images{};
  std::size_t quality_images{};
  std::size_t selected_images{};
  std::size_t used_images{};
  double rms{};
  double cross_validation_rms{};
  double per_view_p95{};
  bool quality_passed{};
  std::vector<std::string> warnings;
};

// Batch-calibrates a pinhole camera from camera_capture saver images.
// The pattern dimensions
// are OpenCV inner-corner counts: an 11x8-square checkerboard is 10x7 here.
// A unique result directory is created below options.output_root.
CameraCalibratorResult calibrateCameraImages(
  const CameraCalibratorOptions& options);

}  // namespace tools
