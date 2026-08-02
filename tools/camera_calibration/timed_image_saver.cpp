#include "tools/camera_calibration/timed_image_saver.hpp"

#include <cmath>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <opencv2/imgcodecs.hpp>

namespace tools {
namespace {

std::string sessionTimestamp()
{
  const std::time_t now = std::time(nullptr);
  std::tm local_time{};
#if defined(_WIN32)
  if (localtime_s(&local_time, &now) != 0) {
    throw std::runtime_error("failed to convert session timestamp");
  }
#else
  if (localtime_r(&now, &local_time) == nullptr) {
    throw std::runtime_error("failed to convert session timestamp");
  }
#endif

  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");
  return stream.str();
}

std::filesystem::path createUniqueSessionDirectory(
  const std::filesystem::path& output_dir)
{
  std::error_code error;
  std::filesystem::create_directories(output_dir, error);
  if (error) {
    throw std::filesystem::filesystem_error(
      "failed to create image output directory", output_dir, error);
  }

  if (!std::filesystem::is_directory(output_dir, error)) {
    if (error) {
      throw std::filesystem::filesystem_error(
        "failed to inspect image output directory", output_dir, error);
    }
    throw std::runtime_error(
      "image output path is not a directory: " + output_dir.string());
  }

  const std::string base_name = sessionTimestamp();
  for (std::uint64_t suffix = 0;; ++suffix) {
    std::string candidate_name = base_name;
    if (suffix > 0) {
      candidate_name += "_" + std::to_string(suffix);
    }

    const auto candidate = output_dir / candidate_name;
    error.clear();
    if (std::filesystem::create_directory(candidate, error)) {
      return candidate;
    }
    if (error) {
      throw std::filesystem::filesystem_error(
        "failed to create image session directory", candidate, error);
    }
  }
}

std::filesystem::path nextImagePath(
  const std::filesystem::path& session_directory,
  std::uint64_t& next_file_index)
{
  while (true) {
    std::ostringstream file_name;
    file_name << std::setfill('0') << std::setw(6)
              << next_file_index << ".png";
    const auto candidate = session_directory / file_name.str();

    std::error_code error;
    const bool exists = std::filesystem::exists(candidate, error);
    if (error) {
      throw std::filesystem::filesystem_error(
        "failed to inspect image output path", candidate, error);
    }
    if (!exists) {
      return candidate;
    }
    ++next_file_index;
  }
}

}  // namespace

TimedImageSaver::TimedImageSaver(TimedImageSaverConfig config)
: config_(std::move(config))
{
  if (!std::isfinite(config_.fps) || config_.fps <= 0.0) {
    throw std::invalid_argument(
      "image capture fps must be a finite positive number");
  }
  if (config_.output_dir.empty()) {
    throw std::invalid_argument("image capture output directory is empty");
  }

  session_directory_ = createUniqueSessionDirectory(config_.output_dir);
}

bool TimedImageSaver::save(
  const cv::Mat& image,
  std::chrono::steady_clock::time_point timestamp)
{
  if (image.empty()) {
    return false;
  }

  if (last_observed_timestamp_
      && timestamp <= *last_observed_timestamp_) {
    return false;
  }
  last_observed_timestamp_ = timestamp;

  if (last_saved_timestamp_) {
    const double elapsed_seconds = std::chrono::duration<double>(
      timestamp - *last_saved_timestamp_).count();
    if (elapsed_seconds < 1.0 / config_.fps) {
      return false;
    }
  }

  const auto image_path =
    nextImagePath(session_directory_, next_file_index_);
  try {
    if (!cv::imwrite(image_path.string(), image)) {
      throw std::runtime_error(
        "OpenCV failed to write PNG: " + image_path.string());
    }
  } catch (const cv::Exception& error) {
    throw std::runtime_error(
      "failed to write PNG '" + image_path.string()
      + "': " + error.what());
  }

  last_saved_timestamp_ = timestamp;
  ++next_file_index_;
  ++saved_count_;
  return true;
}

const std::filesystem::path&
TimedImageSaver::sessionDirectory() const noexcept
{
  return session_directory_;
}

std::uint64_t TimedImageSaver::savedCount() const noexcept
{
  return saved_count_;
}

}  // namespace tools
