#include "tools/record/record_writer.hpp"

#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>

namespace tools {

RecordWriter::RecordWriter(std::filesystem::path base_path, double fps)
: base_path_(std::move(base_path)), fps_(fps)
{
  if (base_path_.empty() || base_path_.has_extension()) {
    throw std::invalid_argument("output must be a base path without .avi/.txt");
  }
  if (!std::isfinite(fps_) || fps_ <= 0.0) {
    throw std::invalid_argument("fps must be positive and finite");
  }
  if (std::filesystem::exists(base_path_.string() + ".avi") ||
      std::filesystem::exists(base_path_.string() + ".txt")) {
    throw std::runtime_error("recording already exists: " + base_path_.string());
  }
}

RecordWriter::~RecordWriter()
{
  close();
}

void RecordWriter::open(const cv::Size& size)
{
  const auto video_path = base_path_.string() + ".avi";
  const auto text_path = base_path_.string() + ".txt";
  if (std::filesystem::exists(video_path) || std::filesystem::exists(text_path)) {
    throw std::runtime_error("recording already exists: " + base_path_.string());
  }
  if (!base_path_.parent_path().empty()) {
    std::filesystem::create_directories(base_path_.parent_path());
  }

  video_.open(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps_, size);
  if (!video_.isOpened()) {
    throw std::runtime_error("cannot open video writer: " + video_path);
  }
  text_.open(text_path, std::ios::out | std::ios::trunc);
  if (!text_) {
    video_.release();
    throw std::runtime_error("cannot open pose text: " + text_path);
  }
  text_ << std::fixed << std::setprecision(9);
  image_size_ = size;
}

void RecordWriter::write(
  const cv::Mat& image, const Eigen::Quaterniond& q_world_imu,
  std::chrono::steady_clock::time_point timestamp)
{
  if (image.empty() || image.type() != CV_8UC3) {
    throw std::invalid_argument("record image must be nonempty BGR CV_8UC3");
  }
  if (!q_world_imu.coeffs().allFinite() ||
      std::abs(q_world_imu.squaredNorm() - 1.0) > 1e-3) {
    throw std::invalid_argument("record quaternion must be finite and normalized");
  }
  if (frames_ == 0) {
    if (!video_.isOpened()) {
      open(image.size());
    }
    first_timestamp_ = timestamp;
  } else if (timestamp <= last_timestamp_) {
    throw std::invalid_argument("record timestamps must increase");
  }
  if (image.size() != image_size_) {
    throw std::invalid_argument("record image size changed");
  }

  // 不给视频压时间戳：AVI/MJPG 是固定帧率，真实图像时刻存在同下标 TXT 中。
  video_.write(image);
  if (!video_.isOpened()) {
    throw std::runtime_error("video writer closed while recording");
  }
  const double seconds =
    std::chrono::duration<double>(timestamp - first_timestamp_).count();
  text_ << seconds << ' ' << q_world_imu.w() << ' ' << q_world_imu.x()
        << ' ' << q_world_imu.y() << ' ' << q_world_imu.z() << '\n';
  if (!text_) {
    throw std::runtime_error("failed to write pose text");
  }
  last_timestamp_ = timestamp;
  ++frames_;
}

void RecordWriter::close()
{
  if (video_.isOpened()) {
    video_.release();
  }
  if (text_.is_open()) {
    text_.close();
  }
}

}  // namespace tools
