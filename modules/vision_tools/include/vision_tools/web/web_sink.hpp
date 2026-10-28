#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace L6Telemetry {

// 远程 Web 调试的旁路出口：标注帧写共享内存、时序写 JSON，
// 供 web.py 转成 MJPEG / Chart.js。与 UdpJsonSender(PlotJuggler) 并列，
// 两边消费同一份 telemetryFrame 字段，不另起 schema。
//
// 只在 debug.web 打开时构造；整体删除不影响瞄准链路。
struct WebSinkStats {
  std::uint64_t submitted_frames{};
  std::uint64_t encoded_frames{};
  std::uint64_t dropped_frames{};
  double encode_fps{};
  double encode_ms{};
};

class WebSink {
public:
  // max_samples: 时序列缓冲长度。前端可再裁，这里限制的是 shm/磁盘占用。
  // record_dir 非空时额外把每条样本追加到 <record_dir>/series.jsonl，
  // 每张帧写成 <record_dir>/frames/frame_%06d.jpg，供赛后回放。
  // record_max_frames <=0 表示不限；达到上限后只停帧录制，时序仍继续。
  explicit WebSink(
    int max_samples = 200, std::string record_dir = {},
    int record_max_frames = 3000,
    std::string ipc_prefix = "/dev/shm/newvision");
  ~WebSink();

  WebSink(const WebSink&) = delete;
  WebSink& operator=(const WebSink&) = delete;
  WebSink(WebSink&&) = delete;
  WebSink& operator=(WebSink&&) = delete;

  // 只复制并提交最新标注帧；后台线程缩放、压 JPEG 后写共享内存。
  // 队列满时覆盖尚未编码的旧帧，调试网页不能反压视觉主循环。
  void writeFrame(const cv::Mat& annotated, std::uint64_t source_frame = 0,
                  double capture_seconds = 0.0);

  WebSinkStats stats() const;

  // 一帧 telemetryFrame 嵌套 JSON → 拍平成 "gimbal/yaw" 这类序列，
  // 追加到环形缓冲；共享文件最多 5 Hz 刷新，避免调试输出抢占主循环。
  void writeSeries(const nlohmann::json& sample);

  // 立即把当前时序缓冲落盘。测试、关停或需要同步读取时使用；正常运行
  // 无需逐帧调用。
  void flush();

  // 结构化日志 → /dev/shm/newvision_log.json。key 相同则覆盖该子树。
  void writeLog(const std::string& key, const nlohmann::json& value);

  // 追加一条事件到 log.events（环形，最多 50 条）。type/detail 会原样落在
  // 前端 Log 树里，供"刚才发生了什么"快速对时间线。
  void writeEvent(const std::string& type, const nlohmann::json& detail = {});

private:
  void frameWorkerLoop();
  void encodeAndPublishFrame(cv::Mat frame, std::uint64_t source_frame,
                             double capture_seconds);
  void flushSeriesLocked();
  void flushLogLocked();

  int max_samples_{200};
  std::mutex mutex_;

  // 平行序列：keys_[i] 的样本在 series_[i]，time_ 为横轴（秒）。
  std::vector<std::string> keys_;
  std::unordered_set<std::string> known_keys_;
  std::vector<std::vector<double>> series_;
  std::vector<double> time_;
  std::chrono::steady_clock::time_point last_series_flush_{};
  nlohmann::json log_ = nlohmann::json::object();

  int frame_fd_{-1};
  void* frame_map_{nullptr};
  std::size_t frame_map_size_{0};

  std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  std::condition_variable frame_drained_cv_;
  cv::Mat pending_frame_;
  std::uint64_t pending_source_frame_{0};
  double pending_capture_seconds_{0.0};
  bool frame_pending_{false};
  bool frame_encoding_{false};
  bool frame_stop_{false};
  std::thread frame_worker_;
  std::atomic<std::uint64_t> submitted_frames_{0};
  std::atomic<std::uint64_t> encoded_frames_{0};
  std::atomic<std::uint64_t> dropped_frames_{0};
  std::atomic<double> encode_fps_{0.0};
  std::atomic<double> encode_ms_{0.0};

  std::string record_dir_;
  std::string ipc_prefix_;
  std::size_t frame_seq_{0};
  int record_max_frames_{0};
  void* series_file_{nullptr};
  void* frame_index_file_{nullptr};
};

}  // namespace L6Telemetry
