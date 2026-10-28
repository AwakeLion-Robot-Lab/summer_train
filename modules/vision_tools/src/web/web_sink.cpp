#include "vision_tools/web/web_sink.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <utility>

namespace L6Telemetry {
namespace {

constexpr std::size_t kFrameMapSize = 2 * 1024 * 1024;
constexpr auto kSeriesFlushInterval = std::chrono::milliseconds(200);
constexpr int kFrameMaxWidth = 960;
constexpr int kFrameMaxHeight = 720;

void flattenSample(
  const nlohmann::json& node, const std::string& prefix,
  std::map<std::string, double>& out)
{
  if (node.is_object()) {
    for (auto it = node.begin(); it != node.end(); ++it) {
      const std::string key = prefix.empty() ? it.key() : prefix + "/" + it.key();
      flattenSample(it.value(), key, out);
    }
    return;
  }
  if (node.is_number()) {
    out.emplace(prefix, node.get<double>());
    return;
  }
  if (node.is_boolean()) {
    out.emplace(prefix, node.get<bool>() ? 1.0 : 0.0);
  }
  // 数组/字符串不进曲线：PlotJuggler 同样只用标量分支。
}

bool writeJsonAtomic(const char* path, const nlohmann::json& value)
{
  const std::string tmp = std::string(path) + ".tmp";
  {
    std::FILE* file = std::fopen(tmp.c_str(), "wb");
    if (!file) {
      return false;
    }
    const std::string dumped = value.dump();
    const auto written = std::fwrite(dumped.data(), 1, dumped.size(), file);
    std::fclose(file);
    if (written != dumped.size()) {
      return false;
    }
  }
  return std::rename(tmp.c_str(), path) == 0;
}

}  // namespace

WebSink::WebSink(int max_samples, std::string record_dir, int record_max_frames,
                 std::string ipc_prefix)
  : max_samples_(max_samples > 0 ? max_samples : 200),
    record_dir_(std::move(record_dir)),
    ipc_prefix_(std::move(ipc_prefix)),
    record_max_frames_(record_max_frames)
{
  frame_fd_ = ::open((ipc_prefix_ + "_frame").c_str(), O_CREAT | O_RDWR, 0666);
  if (frame_fd_ >= 0) {
    if (::ftruncate(frame_fd_, static_cast<off_t>(kFrameMapSize)) == 0) {
      void* mapped = ::mmap(
        nullptr, kFrameMapSize, PROT_READ | PROT_WRITE, MAP_SHARED, frame_fd_, 0);
      if (mapped != MAP_FAILED) {
        frame_map_ = mapped;
        frame_map_size_ = kFrameMapSize;
      }
    }
  }

  if (!record_dir_.empty()) {
    std::error_code error;
    std::filesystem::create_directories(record_dir_, error);
    std::filesystem::create_directories(record_dir_ + "/frames", error);
    series_file_ = std::fopen((record_dir_ + "/series.jsonl").c_str(), "ab");
    frame_index_file_ = std::fopen((record_dir_ + "/frames/index.jsonl").c_str(), "ab");
  }

  if (frame_map_) {
    frame_worker_ = std::thread(&WebSink::frameWorkerLoop, this);
  }
}

WebSink::~WebSink()
{
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_stop_ = true;
  }
  frame_cv_.notify_one();
  if (frame_worker_.joinable()) {
    frame_worker_.join();
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!time_.empty()) {
      flushSeriesLocked();
    }
  }
  if (series_file_) {
    std::fclose(static_cast<std::FILE*>(series_file_));
    series_file_ = nullptr;
  }
  if (frame_index_file_) {
    std::fclose(static_cast<std::FILE*>(frame_index_file_));
    frame_index_file_ = nullptr;
  }
  if (frame_map_) {
    ::munmap(frame_map_, frame_map_size_);
    frame_map_ = nullptr;
  }
  if (frame_fd_ >= 0) {
    ::close(frame_fd_);
    frame_fd_ = -1;
  }
}

void WebSink::writeFrame(const cv::Mat& annotated, std::uint64_t source_frame,
                         double capture_seconds)
{
  if (!frame_map_ || annotated.empty()) {
    return;
  }

  cv::Mat copy = annotated.clone();
  if (copy.empty()) {
    return;
  }

  submitted_frames_.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (frame_pending_) {
      dropped_frames_.fetch_add(1, std::memory_order_relaxed);
    }
    pending_frame_ = std::move(copy);
    pending_source_frame_ = source_frame;
    pending_capture_seconds_ = capture_seconds;
    frame_pending_ = true;
  }
  frame_cv_.notify_one();
}

WebSinkStats WebSink::stats() const
{
  return {
    submitted_frames_.load(std::memory_order_relaxed),
    encoded_frames_.load(std::memory_order_relaxed),
    dropped_frames_.load(std::memory_order_relaxed),
    encode_fps_.load(std::memory_order_relaxed),
    encode_ms_.load(std::memory_order_relaxed)};
}

void WebSink::frameWorkerLoop()
{
  auto rate_started = std::chrono::steady_clock::now();
  std::uint64_t rate_encoded = encoded_frames_.load(std::memory_order_relaxed);

  while (true) {
    cv::Mat frame;
    std::uint64_t source_frame = 0;
    double capture_seconds = 0.0;
    {
      std::unique_lock<std::mutex> lock(frame_mutex_);
      frame_cv_.wait(lock, [this] { return frame_stop_ || frame_pending_; });
      if (frame_stop_ && !frame_pending_) {
        break;
      }
      frame = std::move(pending_frame_);
      source_frame = pending_source_frame_;
      capture_seconds = pending_capture_seconds_;
      frame_pending_ = false;
      frame_encoding_ = true;
    }

    try {
      encodeAndPublishFrame(std::move(frame), source_frame, capture_seconds);
    } catch (const cv::Exception&) {
      // 调试输出失败不能终止视觉主程序；下一帧仍可继续尝试。
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - rate_started).count();
    if (elapsed >= 1.0) {
      const auto encoded = encoded_frames_.load(std::memory_order_relaxed);
      encode_fps_.store(
        static_cast<double>(encoded - rate_encoded) / elapsed,
        std::memory_order_relaxed);
      rate_encoded = encoded;
      rate_started = now;
    }

    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      frame_encoding_ = false;
      if (!frame_pending_) {
        frame_drained_cv_.notify_all();
      }
    }
  }

  std::lock_guard<std::mutex> lock(frame_mutex_);
  frame_encoding_ = false;
  frame_drained_cv_.notify_all();
}

void WebSink::encodeAndPublishFrame(cv::Mat frame, std::uint64_t source_frame,
                                    double capture_seconds)
{
  const auto encode_started = std::chrono::steady_clock::now();
  cv::Mat output;
  if (frame.cols > kFrameMaxWidth || frame.rows > kFrameMaxHeight) {
    const double scale = std::min(
      static_cast<double>(kFrameMaxWidth) / frame.cols,
      static_cast<double>(kFrameMaxHeight) / frame.rows);
    cv::resize(frame, output, {}, scale, scale, cv::INTER_AREA);
  } else {
    output = std::move(frame);
  }

  // JPEG q75 对标注文字仍足够清晰，同时显著降低无线网络带宽。
  static const std::vector<int> kJpegParams = {cv::IMWRITE_JPEG_QUALITY, 75};
  std::vector<uchar> buffer;
  if (!cv::imencode(".jpg", output, buffer, kJpegParams) ||
      buffer.size() + sizeof(std::uint32_t) > frame_map_size_) {
    return;
  }

  encode_ms_.store(
    std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - encode_started).count(),
    std::memory_order_relaxed);

  // 先把长度清零，写完 JPEG 后再发布新长度。浏览器不会读到半张图。
  const std::uint32_t empty = 0;
  const std::uint32_t size = static_cast<std::uint32_t>(buffer.size());
  std::memcpy(frame_map_, &empty, sizeof(empty));
  std::atomic_thread_fence(std::memory_order_release);
  std::memcpy(
    static_cast<char*>(frame_map_) + sizeof(size), buffer.data(), buffer.size());
  std::atomic_thread_fence(std::memory_order_release);
  std::memcpy(frame_map_, &size, sizeof(size));
  encoded_frames_.fetch_add(1, std::memory_order_relaxed);

  if (!record_dir_.empty()) {
    if (record_max_frames_ > 0 && static_cast<int>(frame_seq_) >= record_max_frames_) {
      return;
    }
    char name[64];
    const std::size_t recorded_index = frame_seq_;
    std::snprintf(name, sizeof(name), "/frames/frame_%06zu.jpg", recorded_index);
    const std::string path = record_dir_ + name;
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file) {
      const bool written = std::fwrite(buffer.data(), 1, buffer.size(), file) == buffer.size();
      const bool closed = std::fclose(file) == 0;
      if (written && closed) {
        ++frame_seq_;
      }
      if (written && closed && frame_index_file_) {
        const nlohmann::json entry = {
          {"index", recorded_index}, {"source_frame", source_frame},
          {"t", capture_seconds}};
        const std::string line = entry.dump() + "\n";
        std::fwrite(line.data(), 1, line.size(), static_cast<std::FILE*>(frame_index_file_));
        std::fflush(static_cast<std::FILE*>(frame_index_file_));
      }
    }
  }
}

void WebSink::writeSeries(const nlohmann::json& sample)
{
  std::map<std::string, double> flat;
  flattenSample(sample, "", flat);

  const auto now_point = std::chrono::steady_clock::now();
  const auto now = std::chrono::duration<double>(now_point.time_since_epoch()).count();

  std::lock_guard<std::mutex> lock(mutex_);

  // "t" 由 telemetryFrame 给出曝光时刻；没有就退回写入时刻。
  double time_value = now;
  const auto time_it = flat.find("t");
  if (time_it != flat.end()) {
    time_value = time_it->second;
    flat.erase(time_it);
  }

  // 目标尚未建立时，第一帧不会包含 aim/、ekf/ 等条件字段。字段表必须
  // 动态扩展，否则这些关键曲线在本次进程余下时间里都不会出现在网页。
  for (const auto& [key, value] : flat) {
    (void)value;
    if (!known_keys_.insert(key).second) {
      continue;
    }
    keys_.push_back(key);
    series_.emplace_back(time_.size(), 0.0);
    series_.back().reserve(static_cast<std::size_t>(max_samples_));
  }

  time_.push_back(time_value);
  for (std::size_t index = 0; index < keys_.size(); ++index) {
    const auto it = flat.find(keys_[index]);
    series_[index].push_back(it == flat.end() ? 0.0 : it->second);
  }

  while (static_cast<int>(time_.size()) > max_samples_) {
    time_.erase(time_.begin());
    for (auto& values : series_) {
      values.erase(values.begin());
    }
  }

  // 浏览器最多需要数 Hz 的曲线刷新。样本仍逐帧保留，只限制整份 JSON 的
  // 序列化与原子落盘频率，避免 Web 调试反过来抢占视觉主循环。
  if (last_series_flush_.time_since_epoch().count() == 0 ||
      now_point - last_series_flush_ >= kSeriesFlushInterval) {
    flushSeriesLocked();
    last_series_flush_ = now_point;
  }

  if (series_file_) {
    nlohmann::json row = sample;
    row["t"] = time_value;
    const std::string line = row.dump() + "\n";
    std::fwrite(line.data(), 1, line.size(), static_cast<std::FILE*>(series_file_));
    std::fflush(static_cast<std::FILE*>(series_file_));
  }
}

void WebSink::flush()
{
  {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    frame_drained_cv_.wait(
      lock, [this] { return !frame_pending_ && !frame_encoding_; });
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!time_.empty()) {
    flushSeriesLocked();
    last_series_flush_ = std::chrono::steady_clock::now();
  }
}

void WebSink::writeLog(const std::string& key, const nlohmann::json& value)
{
  std::lock_guard<std::mutex> lock(mutex_);
  log_[key] = value;
  flushLogLocked();
}

void WebSink::writeEvent(const std::string& type, const nlohmann::json& detail)
{
  nlohmann::json event;
  event["t"] = std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
  event["type"] = type;
  if (!detail.is_null()) {
    event["detail"] = detail;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (!log_["events"].is_array()) {
    log_["events"] = nlohmann::json::array();
  }
  log_["events"].push_back(std::move(event));
  while (log_["events"].size() > 50) {
    log_["events"].erase(log_["events"].begin());
  }
  flushLogLocked();
}

void WebSink::flushSeriesLocked()
{
  nlohmann::json data;
  data["time"] = time_;
  for (std::size_t index = 0; index < keys_.size(); ++index) {
    data[keys_[index]] = series_[index];
  }
  writeJsonAtomic((ipc_prefix_ + "_data.json").c_str(), data);
}

void WebSink::flushLogLocked()
{
  writeJsonAtomic((ipc_prefix_ + "_log.json").c_str(), log_);
}

}  // namespace L6Telemetry
