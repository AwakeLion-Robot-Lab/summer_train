#include "tools/camera_calibration/timed_image_saver.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

class TemporaryDirectory {
public:
  TemporaryDirectory()
  {
    const auto unique = std::chrono::high_resolution_clock::now()
      .time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path()
      / ("newvision_timed_image_saver_smoke_" + std::to_string(unique));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory()
  {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept
  {
    return path_;
  }

private:
  std::filesystem::path path_;
};

int fail(int code, const std::string& message)
{
  std::cerr << message << '\n';
  return code;
}

bool sameImage(const cv::Mat& actual, const cv::Mat& expected)
{
  return !actual.empty()
         && actual.size() == expected.size()
         && actual.type() == expected.type()
         && cv::norm(actual, expected, cv::NORM_INF) == 0.0;
}

template <typename Function>
bool throwsException(Function&& function)
{
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main()
{
  TemporaryDirectory temporary_directory;

  if (!throwsException([&] {
        tools::TimedImageSaver saver{{
          .fps = 0.0,
          .output_dir = temporary_directory.path() / "zero_fps"}};
      })) {
    return fail(1, "Zero FPS was accepted");
  }
  if (!throwsException([&] {
        tools::TimedImageSaver saver{{
          .fps = std::numeric_limits<double>::quiet_NaN(),
          .output_dir = temporary_directory.path() / "nan_fps"}};
      })) {
    return fail(2, "NaN FPS was accepted");
  }
  if (!throwsException([] {
        tools::TimedImageSaver saver{{
          .fps = 1.0,
          .output_dir = {}}};
      })) {
    return fail(3, "Empty output directory was accepted");
  }

  const auto regular_file = temporary_directory.path() / "not_a_directory";
  {
    std::ofstream output{regular_file};
    output << "not a directory";
  }
  if (!throwsException([&] {
        tools::TimedImageSaver saver{{
          .fps = 1.0,
          .output_dir = regular_file}};
      })) {
    return fail(4, "A regular file was accepted as the output directory");
  }

  const auto output_root = temporary_directory.path() / "calibration_images";
  tools::TimedImageSaver saver{{
    .fps = 2.0,
    .output_dir = output_root}};

  const cv::Mat frame_a(24, 32, CV_8UC3, cv::Scalar{10, 20, 30});
  const cv::Mat frame_b(24, 32, CV_8UC3, cv::Scalar{40, 50, 60});
  const cv::Mat frame_c(24, 32, CV_8UC3, cv::Scalar{70, 80, 90});
  const auto start = Clock::time_point{};

  if (saver.save(cv::Mat{}, start)) {
    return fail(5, "Empty image was saved");
  }
  if (!saver.save(frame_a, start)) {
    return fail(6, "First valid image was not saved immediately");
  }
  if (saver.save(frame_b, start)) {
    return fail(7, "Duplicate timestamp was accepted");
  }
  if (saver.save(frame_b, start + 250ms)) {
    return fail(8, "FPS rate limiting was not applied");
  }
  if (saver.save(frame_b, start + 200ms)) {
    return fail(9, "Regressive timestamp was accepted");
  }
  if (!saver.save(frame_c, start + 500ms)) {
    return fail(10, "Image at the configured interval was rejected");
  }
  if (saver.savedCount() != 2) {
    return fail(11, "Saved image count mismatch");
  }

  const auto first_path = saver.sessionDirectory() / "000001.png";
  const auto second_path = saver.sessionDirectory() / "000002.png";
  if (!sameImage(cv::imread(first_path.string(), cv::IMREAD_UNCHANGED), frame_a)
      || !sameImage(
        cv::imread(second_path.string(), cv::IMREAD_UNCHANGED), frame_c)) {
    return fail(12, "Saved PNG does not match its source image");
  }

  tools::TimedImageSaver second_saver{{
    .fps = 1.0,
    .output_dir = output_root}};
  if (second_saver.sessionDirectory() == saver.sessionDirectory()) {
    return fail(13, "A second saver reused an existing session directory");
  }
  if (!second_saver.save(frame_b, start)
      || !std::filesystem::exists(
        second_saver.sessionDirectory() / "000001.png")) {
    return fail(14, "Second session failed to save its first image");
  }

  std::cout << "Timed image saver smoke test passed\n";
  return 0;
}
