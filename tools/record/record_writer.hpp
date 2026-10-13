#pragma once

#include <Eigen/Geometry>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace tools {

// auto_aim_test 的同名 AVI + TXT 格式。每写入一帧视频，紧接着写对应的一行姿态。
class RecordWriter {
public:
  RecordWriter(std::filesystem::path base_path, double fps);
  ~RecordWriter();

  RecordWriter(const RecordWriter&) = delete;
  RecordWriter& operator=(const RecordWriter&) = delete;

  void write(const cv::Mat& image, const Eigen::Quaterniond& q_world_imu,
             std::chrono::steady_clock::time_point timestamp);
  void close();
  std::size_t frames() const noexcept { return frames_; }

private:
  void open(const cv::Size& size);

  std::filesystem::path base_path_;
  double fps_;
  cv::VideoWriter video_;
  std::ofstream text_;
  cv::Size image_size_;
  std::chrono::steady_clock::time_point first_timestamp_{};
  std::chrono::steady_clock::time_point last_timestamp_{};
  std::size_t frames_{0};
};

}  // namespace tools
