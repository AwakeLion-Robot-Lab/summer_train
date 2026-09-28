#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>

#include <opencv2/core/mat.hpp>

namespace tools {

// Rate-limited lossless saver used by the calibration capture tool.
struct TimedImageSaverConfig {
  double fps = 1.0;
  std::filesystem::path output_dir = "calibration_images";
};

class TimedImageSaver {
public:
  explicit TimedImageSaver(TimedImageSaverConfig config);

  // 时间戳有效、到达保存间隔且 PNG 成功落盘时返回 true。
  // 空图像、重复/倒退时间戳或尚未到达保存间隔时返回 false。
  bool save(
    const cv::Mat& image,
    std::chrono::steady_clock::time_point timestamp);

  [[nodiscard]] const std::filesystem::path&
  sessionDirectory() const noexcept;
  [[nodiscard]] std::uint64_t savedCount() const noexcept;

private:
  TimedImageSaverConfig config_;
  std::filesystem::path session_directory_;
  std::optional<std::chrono::steady_clock::time_point>
    last_observed_timestamp_;
  std::optional<std::chrono::steady_clock::time_point>
    last_saved_timestamp_;
  std::uint64_t next_file_index_ = 1;
  std::uint64_t saved_count_ = 0;
};

}  // namespace tools
