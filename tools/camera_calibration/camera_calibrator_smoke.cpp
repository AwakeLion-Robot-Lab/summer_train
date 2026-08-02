#include "l1_sensor/camera/camera_calibration.hpp"
#include "tools/camera_calibration/high_precision_calibrator.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace {

class TemporaryDirectory {
public:
  TemporaryDirectory()
  {
    const auto unique = std::chrono::high_resolution_clock::now()
      .time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path()
      / ("newvision_camera_calibrator_smoke_"
         + std::to_string(unique));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory()
  {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const noexcept
  {
    return path_;
  }

private:
  std::filesystem::path path_;
};

cv::Mat rotationFromEuler(double x, double y, double z)
{
  const cv::Matx33d rx{
    1.0, 0.0, 0.0,
    0.0, std::cos(x), -std::sin(x),
    0.0, std::sin(x), std::cos(x)};
  const cv::Matx33d ry{
    std::cos(y), 0.0, std::sin(y),
    0.0, 1.0, 0.0,
    -std::sin(y), 0.0, std::cos(y)};
  const cv::Matx33d rz{
    std::cos(z), -std::sin(z), 0.0,
    std::sin(z), std::cos(z), 0.0,
    0.0, 0.0, 1.0};
  const cv::Mat rotation = cv::Mat(rz * ry * rx);
  cv::Mat rotation_vector;
  cv::Rodrigues(rotation, rotation_vector);
  return rotation_vector;
}

cv::Mat makeCheckerboardWithMargin(int square_pixels)
{
  // The board has 11x8 squares and therefore 10x7 inner corners.
  // One white square of margin is included on every side.
  cv::Mat board(
    10 * square_pixels, 13 * square_pixels,
    CV_8UC1, cv::Scalar{255});
  for (int row = 0; row < 8; ++row) {
    for (int column = 0; column < 11; ++column) {
      if ((row + column) % 2 == 0) {
        cv::rectangle(
          board,
          {(column + 1) * square_pixels,
           (row + 1) * square_pixels},
          {(column + 2) * square_pixels - 1,
           (row + 2) * square_pixels - 1},
          cv::Scalar{0}, cv::FILLED);
      }
    }
  }
  return board;
}

void writeSyntheticDataset(
  const std::filesystem::path& directory,
  const cv::Mat& camera_matrix)
{
  std::filesystem::create_directories(directory);
  constexpr int square_pixels = 70;
  const cv::Mat source = makeCheckerboardWithMargin(square_pixels);
  const std::vector<cv::Point2f> source_quad{
    {0.0F, 0.0F},
    {static_cast<float>(source.cols - 1), 0.0F},
    {static_cast<float>(source.cols - 1),
     static_cast<float>(source.rows - 1)},
    {0.0F, static_cast<float>(source.rows - 1)}};
  const std::vector<cv::Point3f> board_with_margin{
    {-1.0F, -1.0F, 0.0F},
    {12.0F, -1.0F, 0.0F},
    {12.0F, 9.0F, 0.0F},
    {-1.0F, 9.0F, 0.0F}};

  for (int index = 0; index < 24; ++index) {
    const double phase =
      2.0 * CV_PI * static_cast<double>(index) / 24.0;
    const cv::Mat rotation_vector = rotationFromEuler(
      0.24 * std::sin(phase * 1.7),
      0.32 * std::cos(phase * 1.3),
      0.30 * std::sin(phase));
    const cv::Mat translation_vector = (cv::Mat_<double>(3, 1)
      << -5.0 + 2.4 * std::sin(phase),
         -3.5 + 1.8 * std::cos(phase * 1.2),
         19.0 + 3.0 * std::sin(phase * 0.7));
    std::vector<cv::Point2f> destination_quad;
    cv::projectPoints(
      board_with_margin, rotation_vector, translation_vector,
      camera_matrix, cv::Mat::zeros(1, 5, CV_64FC1),
      destination_quad);
    const cv::Mat homography =
      cv::getPerspectiveTransform(source_quad, destination_quad);
    cv::Mat image(960, 1280, CV_8UC1, cv::Scalar{235});
    cv::warpPerspective(
      source, image, homography, image.size(),
      cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);

    std::ostringstream filename;
    filename << std::setfill('0') << std::setw(6)
             << index + 1 << ".png";
    if (!cv::imwrite((directory / filename.str()).string(), image)) {
      throw std::runtime_error("failed to write synthetic calibration image");
    }
  }
}

int fail(int code, const std::string& message)
{
  std::cerr << message << '\n';
  return code;
}

}  // namespace

int main()
{
  TemporaryDirectory temporary_directory;
  const cv::Mat expected_camera_matrix = (cv::Mat_<double>(3, 3)
    << 1030.0, 0.0, 638.0,
       0.0, 1015.0, 482.0,
       0.0, 0.0, 1.0);
  const auto image_directory =
    temporary_directory.path() / "images";
  writeSyntheticDataset(image_directory, expected_camera_matrix);

  tools::CameraCalibratorOptions options;
  options.input_directory = image_directory;
  options.output_root =
    temporary_directory.path() / "results";
  options.bootstrap_iterations = 2;
  options.max_sharpness = 5.0;
  options.min_views = 15;
  options.max_views = 24;
  const auto result = tools::calibrateCameraImages(options);

  if (result.discovered_images != 24
      || result.detected_images < 20
      || result.used_images < 15) {
    return fail(1, "Synthetic checkerboard detection count is too low");
  }
  if (result.image_size != cv::Size{1280, 960}) {
    return fail(2, "Synthetic calibration image size is incorrect");
  }
  const double fx_relative_error = std::abs(
    result.camera_matrix.at<double>(0, 0)
      - expected_camera_matrix.at<double>(0, 0))
    / expected_camera_matrix.at<double>(0, 0);
  const double fy_relative_error = std::abs(
    result.camera_matrix.at<double>(1, 1)
      - expected_camera_matrix.at<double>(1, 1))
    / expected_camera_matrix.at<double>(1, 1);
  if (fx_relative_error > 0.05 || fy_relative_error > 0.05) {
    return fail(3, "Recovered synthetic focal length is outside 5 percent");
  }
  if (result.rms > 0.5) {
    return fail(4, "Synthetic calibration reprojection RMS is too high");
  }

  const auto yaml_path =
    result.output_directory / "calibration.yaml";
  const YAML::Node yaml = YAML::LoadFile(yaml_path.string());
  const auto loaded = L1Sensor::loadCameraCalibration(
    yaml["calibration"], yaml_path.string());
  if (!loaded.matchesImageSize({1280, 960})
      || loaded.distortion_coefficients.total() != 5
      || loaded.barrelExtrinsicsReady()) {
    return fail(5, "Generated YAML is incompatible with runtime loader");
  }
  if (!std::filesystem::exists(
        result.output_directory / "report.yaml")
      || !std::filesystem::exists(
        result.output_directory / "coverage.png")
      || !std::filesystem::exists(
        result.output_directory / "reprojection_errors.png")) {
    return fail(6, "Calibration diagnostics were not generated");
  }

  std::cout << "Camera calibrator smoke test passed\n";
  return 0;
}
