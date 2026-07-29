#include "tools/recorder.hpp"

#include "l6_telemetry/logger.hpp"
#include "yaml.hpp"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#include <opencv2/videoio.hpp>

namespace tools {
namespace {

constexpr std::size_t kQueueCapacity = 8;
constexpr double kDefaultFps = 30.0;
const std::filesystem::path kDefaultOutputDirectory{"records"};

template <typename T>
T readOptional(
  const YAML::Node& node,
  const std::string& key,
  const T& default_value)
{
  if (!node[key]) {
    return default_value;
  }

  try {
    return node[key].as<T>();
  } catch (const YAML::Exception& error) {
    L6Telemetry::logWarn("recorder config invalid field", key, error.what());
    return default_value;
  }
}

std::string sessionTimestamp()
{
  const std::time_t now = std::time(nullptr);
  std::tm local_time{};
#if defined(_WIN32)
  localtime_s(&local_time, &now);
#else
  localtime_r(&now, &local_time);
#endif

  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");
  return stream.str();
}

bool validQuaternion(const Eigen::Quaterniond& quaternion)
{
  const double norm = quaternion.norm();
  return quaternion.coeffs().allFinite()
         && std::isfinite(norm)
         && norm > 1e-12;
}

}  // namespace

RecorderConfig loadRecorderConfig(const std::string& config_path)
{
  RecorderConfig config;
  const YAML::Node root = tools::load(config_path);
  const YAML::Node recorder = root["recorder"];
  if (!recorder) {
    return config;
  }

  if (!recorder.IsMap()) {
    L6Telemetry::logWarn("recorder config must be a map; recorder disabled");
    return config;
  }

  config.enabled = readOptional(recorder, "enabled", config.enabled);
  config.fps = readOptional(recorder, "fps", config.fps);
  config.output_dir =
    readOptional(recorder, "output_dir", config.output_dir.string());

  if (!std::isfinite(config.fps) || config.fps <= 0.0) {
    L6Telemetry::logWarn(
      "recorder config fps invalid; use default", config.fps, kDefaultFps);
    config.fps = kDefaultFps;
  }
  if (config.output_dir.empty()) {
    L6Telemetry::logWarn(
      "recorder config output_dir empty; use default",
      kDefaultOutputDirectory.string());
    config.output_dir = kDefaultOutputDirectory;
  }

  return config;
}

class Recorder::Impl {
public:
  explicit Impl(RecorderConfig config)
  : config_(std::move(config))
  {
    if (!std::isfinite(config_.fps) || config_.fps <= 0.0) {
      L6Telemetry::logWarn(
        "recorder fps invalid; use default", config_.fps, kDefaultFps);
      config_.fps = kDefaultFps;
    }
    if (config_.output_dir.empty()) {
      L6Telemetry::logWarn(
        "recorder output directory empty; use default",
        kDefaultOutputDirectory.string());
      config_.output_dir = kDefaultOutputDirectory;
    }
  }

  ~Impl()
  {
    stop();
  }

  bool record(
    const cv::Mat& image,
    const Eigen::Quaterniond& quaternion,
    std::chrono::steady_clock::time_point timestamp)
  {
    if (!config_.enabled) {
      return false;
    }
    if (image.empty() || image.type() != CV_8UC3
        || !validQuaternion(quaternion)) {
      rejected_count_.fetch_add(1);
      return false;
    }

    const Eigen::Quaterniond normalized = quaternion.normalized();
    std::unique_lock<std::mutex> lock(mutex_);
    if (!accepting_ || failed_) {
      return false;
    }

    if (!initialized_ && !initialize(image)) {
      return false;
    }
    if (image.size() != frame_size_) {
      rejected_count_.fetch_add(1);
      return false;
    }

    if (last_accepted_timestamp_) {
      if (timestamp <= *last_accepted_timestamp_) {
        rejected_count_.fetch_add(1);
        return false;
      }

      const double elapsed = std::chrono::duration<double>(
        timestamp - *last_accepted_timestamp_).count();
      if (elapsed + 1e-9 < 1.0 / config_.fps) {
        rate_limited_count_.fetch_add(1);
        return false;
      }
    }

    if (queue_.size() >= kQueueCapacity) {
      queue_dropped_count_.fetch_add(1);
      return false;
    }

    try {
      queue_.push_back(FrameData{image.clone(), normalized, timestamp});
    } catch (const cv::Exception& error) {
      rejected_count_.fetch_add(1);
      L6Telemetry::logError("recorder image clone failed", error.what());
      return false;
    }

    if (!last_accepted_timestamp_) {
      first_timestamp_ = timestamp;
    }
    last_accepted_timestamp_ = timestamp;
    accepted_count_.fetch_add(1);
    lock.unlock();
    queue_condition_.notify_one();
    return true;
  }

  void stop()
  {
    std::lock_guard<std::mutex> stop_lock(stop_mutex_);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      accepting_ = false;
      stop_requested_ = true;
    }
    queue_condition_.notify_all();

    if (worker_.joinable()) {
      worker_.join();
    }

    if (text_writer_.is_open()) {
      text_writer_.flush();
      text_writer_.close();
    }
    if (video_writer_.isOpened()) {
      video_writer_.release();
    }

    if (initialized_ && !summary_logged_) {
      summary_logged_ = true;
      L6Telemetry::logInfo(
        "recorder stopped",
        "video", video_path_.string(),
        "pose", text_path_.string(),
        "accepted", accepted_count_.load(),
        "written", written_count_.load(),
        "rate_limited", rate_limited_count_.load(),
        "queue_dropped", queue_dropped_count_.load(),
        "rejected", rejected_count_.load());
    }
  }

private:
  struct FrameData {
    cv::Mat image;
    Eigen::Quaterniond quaternion;
    std::chrono::steady_clock::time_point timestamp;
  };

  bool initialize(const cv::Mat& first_image)
  {
    try {
      std::filesystem::create_directories(config_.output_dir);

      const std::string base_name = sessionTimestamp();
      for (std::uint64_t suffix = 0;; ++suffix) {
        std::string candidate = base_name;
        if (suffix > 0) {
          candidate += "_" + std::to_string(suffix);
        }

        const auto video_candidate =
          config_.output_dir / (candidate + ".avi");
        const auto text_candidate =
          config_.output_dir / (candidate + ".txt");
        if (!std::filesystem::exists(video_candidate)
            && !std::filesystem::exists(text_candidate)) {
          video_path_ = video_candidate;
          text_path_ = text_candidate;
          break;
        }
      }

      const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
      if (!video_writer_.open(
            video_path_.string(),
            fourcc,
            config_.fps,
            first_image.size(),
            true)) {
        throw std::runtime_error(
          "failed to open video writer: " + video_path_.string());
      }

      text_writer_.open(text_path_);
      if (!text_writer_.is_open()) {
        throw std::runtime_error(
          "failed to open pose writer: " + text_path_.string());
      }
      text_writer_ << std::fixed << std::setprecision(9);

      frame_size_ = first_image.size();
      initialized_ = true;
      worker_ = std::thread(&Impl::saveLoop, this);
      L6Telemetry::logInfo(
        "recorder started",
        "video", video_path_.string(),
        "pose", text_path_.string(),
        "fps", config_.fps,
        "width", frame_size_.width,
        "height", frame_size_.height);
      return true;
    } catch (const std::exception& error) {
      failed_ = true;
      accepting_ = false;
      if (text_writer_.is_open()) {
        text_writer_.close();
      }
      if (video_writer_.isOpened()) {
        video_writer_.release();
      }

      std::error_code ignored;
      if (!video_path_.empty()) {
        std::filesystem::remove(video_path_, ignored);
      }
      if (!text_path_.empty()) {
        std::filesystem::remove(text_path_, ignored);
      }

      L6Telemetry::logError("recorder initialization failed", error.what());
      return false;
    }
  }

  void saveLoop()
  {
    while (true) {
      FrameData frame;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_condition_.wait(
          lock,
          [this] { return stop_requested_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stop_requested_) {
            break;
          }
          continue;
        }

        frame = std::move(queue_.front());
        queue_.pop_front();
      }

      try {
        video_writer_.write(frame.image);
        const double relative_seconds = std::chrono::duration<double>(
          frame.timestamp - first_timestamp_).count();
        text_writer_
          << relative_seconds << ' '
          << frame.quaternion.w() << ' '
          << frame.quaternion.x() << ' '
          << frame.quaternion.y() << ' '
          << frame.quaternion.z() << '\n';
        if (!text_writer_) {
          throw std::runtime_error("failed to write pose text");
        }
        written_count_.fetch_add(1);
      } catch (const std::exception& error) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          failed_ = true;
          accepting_ = false;
          rejected_count_.fetch_add(queue_.size());
          queue_.clear();
        }
        L6Telemetry::logError("recorder write failed", error.what());
        break;
      }
    }
  }

  RecorderConfig config_;
  std::mutex mutex_;
  std::mutex stop_mutex_;
  std::condition_variable queue_condition_;
  std::deque<FrameData> queue_;
  std::thread worker_;
  bool accepting_ = true;
  bool stop_requested_ = false;
  bool initialized_ = false;
  bool failed_ = false;
  bool summary_logged_ = false;
  cv::Size frame_size_;
  std::chrono::steady_clock::time_point first_timestamp_{};
  std::optional<std::chrono::steady_clock::time_point>
    last_accepted_timestamp_;
  std::filesystem::path video_path_;
  std::filesystem::path text_path_;
  std::ofstream text_writer_;
  cv::VideoWriter video_writer_;
  std::atomic<std::uint64_t> accepted_count_{0};
  std::atomic<std::uint64_t> written_count_{0};
  std::atomic<std::uint64_t> rate_limited_count_{0};
  std::atomic<std::uint64_t> queue_dropped_count_{0};
  std::atomic<std::uint64_t> rejected_count_{0};
};

Recorder::Recorder(RecorderConfig config)
: impl_(std::make_unique<Impl>(std::move(config)))
{
}

Recorder::~Recorder() = default;

bool Recorder::record(
  const cv::Mat& image,
  const Eigen::Quaterniond& quaternion,
  std::chrono::steady_clock::time_point timestamp)
{
  return impl_->record(image, quaternion, timestamp);
}

void Recorder::stop()
{
  impl_->stop();
}

}  // namespace tools
