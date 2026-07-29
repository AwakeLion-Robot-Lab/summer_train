#include "tools/recorder.hpp"

#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct PoseRecord {
  double time_seconds = 0.0;
  Eigen::Quaterniond quaternion = Eigen::Quaterniond::Identity();
};

class TemporaryDirectory {
public:
  TemporaryDirectory()
  {
    const auto unique = std::chrono::high_resolution_clock::now()
      .time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path()
      / ("newvision_recorder_smoke_" + std::to_string(unique));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory()
  {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const
  {
    return path_;
  }

private:
  std::filesystem::path path_;
};

bool close(double actual, double expected, double tolerance = 1e-7)
{
  return std::abs(actual - expected) <= tolerance;
}

std::vector<std::filesystem::path> filesWithExtension(
  const std::filesystem::path& directory,
  const std::string& extension)
{
  std::vector<std::filesystem::path> result;
  if (!std::filesystem::exists(directory)) {
    return result;
  }

  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == extension) {
      result.push_back(entry.path());
    }
  }
  return result;
}

std::vector<PoseRecord> readPoseRecords(const std::filesystem::path& path)
{
  std::ifstream input(path);
  std::vector<PoseRecord> records;
  double time_seconds = 0.0;
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  while (input >> time_seconds >> w >> x >> y >> z) {
    records.push_back(
      PoseRecord{time_seconds, Eigen::Quaterniond{w, x, y, z}});
  }
  return records;
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

  const auto config_path = temporary_directory.path() / "recorder.yaml";
  const auto configured_output = temporary_directory.path() / "configured";
  {
    std::ofstream config_stream(config_path);
    config_stream
      << "recorder:\n"
      << "  enabled: true\n"
      << "  fps: 12.5\n"
      << "  output_dir: " << configured_output.string() << '\n';
  }
  const tools::RecorderConfig loaded =
    tools::loadRecorderConfig(config_path.string());
  if (!loaded.enabled || !close(loaded.fps, 12.5)
      || loaded.output_dir != configured_output) {
    return fail(1, "Recorder config parsing failed");
  }

  const auto disabled_output = temporary_directory.path() / "disabled";
  tools::RecorderConfig disabled_config;
  disabled_config.output_dir = disabled_output;
  {
    tools::Recorder disabled_recorder(disabled_config);
    const cv::Mat frame(24, 32, CV_8UC3, cv::Scalar{10, 20, 30});
    if (disabled_recorder.record(
          frame, Eigen::Quaterniond::Identity(), Clock::time_point{})) {
      return fail(2, "Disabled recorder accepted a frame");
    }
    disabled_recorder.stop();
    disabled_recorder.stop();
  }
  if (std::filesystem::exists(disabled_output)) {
    return fail(3, "Disabled recorder created an output directory");
  }

  const auto output = temporary_directory.path() / "records";
  tools::RecorderConfig config;
  config.enabled = true;
  config.fps = 10.0;
  config.output_dir = output;

  const cv::Mat frame_a(24, 32, CV_8UC3, cv::Scalar{10, 20, 30});
  const cv::Mat frame_b(24, 32, CV_8UC3, cv::Scalar{40, 50, 60});
  const cv::Mat frame_c(24, 32, CV_8UC3, cv::Scalar{70, 80, 90});
  const auto start = Clock::time_point{};
  const Eigen::Quaterniond quarter_turn{
    Eigen::AngleAxisd(std::numbers::pi / 2.0, Eigen::Vector3d::UnitZ())};

  {
    tools::Recorder recorder(config);
    if (!recorder.record(
          frame_a, Eigen::Quaterniond::Identity(), start)) {
      return fail(4, "Recorder rejected the first valid frame");
    }
    if (recorder.record(
          frame_b, Eigen::Quaterniond::Identity(), start + 50ms)) {
      return fail(5, "Recorder did not apply FPS rate limiting");
    }
    if (!recorder.record(
          frame_b, Eigen::Quaterniond{2.0, 0.0, 0.0, 0.0}, start + 100ms)) {
      return fail(6, "Recorder rejected the second valid frame");
    }
    if (recorder.record(
          frame_b, Eigen::Quaterniond::Identity(), start + 100ms)) {
      return fail(7, "Recorder accepted a duplicate timestamp");
    }

    const Eigen::Quaterniond invalid{
      std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0};
    if (recorder.record(frame_c, invalid, start + 200ms)) {
      return fail(8, "Recorder accepted an invalid quaternion");
    }
    if (recorder.record(cv::Mat{}, quarter_turn, start + 200ms)) {
      return fail(9, "Recorder accepted an empty image");
    }
    const cv::Mat wrong_size(12, 16, CV_8UC3, cv::Scalar{1, 2, 3});
    if (recorder.record(wrong_size, quarter_turn, start + 200ms)) {
      return fail(10, "Recorder accepted a resolution change");
    }
    if (!recorder.record(frame_c, quarter_turn, start + 200ms)) {
      return fail(11, "Recorder rejected the third valid frame");
    }

    recorder.stop();
    recorder.stop();
  }

  const auto videos = filesWithExtension(output, ".avi");
  const auto poses = filesWithExtension(output, ".txt");
  if (videos.size() != 1 || poses.size() != 1) {
    return fail(12, "Recorder did not create exactly one AVI/TXT pair");
  }

  cv::VideoCapture video(videos.front());
  if (!video.isOpened()) {
    return fail(13, "Recorded AVI could not be opened");
  }
  if (!close(video.get(cv::CAP_PROP_FPS), 10.0, 0.01)) {
    return fail(14, "Recorded AVI FPS mismatch");
  }
  if (static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)) != frame_a.cols
      || static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT))
           != frame_a.rows) {
    return fail(15, "Recorded AVI resolution mismatch");
  }

  int video_frame_count = 0;
  cv::Mat decoded;
  while (video.read(decoded)) {
    ++video_frame_count;
  }
  const auto records = readPoseRecords(poses.front());
  if (video_frame_count != 3
      || records.size() != static_cast<std::size_t>(video_frame_count)) {
    return fail(16, "Video frame count and pose row count mismatch");
  }

  if (!close(records[0].time_seconds, 0.0)
      || !close(records[1].time_seconds, 0.1)
      || !close(records[2].time_seconds, 0.2)) {
    return fail(17, "Recorded relative timestamps mismatch");
  }
  for (const auto& record : records) {
    if (!close(record.quaternion.norm(), 1.0)) {
      return fail(18, "Recorded quaternion was not normalized");
    }
  }
  if (!close(records[2].quaternion.w(), std::sqrt(0.5), 1e-6)
      || !close(records[2].quaternion.z(), std::sqrt(0.5), 1e-6)) {
    return fail(19, "Recorded quaternion component order mismatch");
  }

  std::cout << "Recorder smoke test passed\n";
  return 0;
}
