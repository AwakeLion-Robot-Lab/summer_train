// WebSink 旁路出口冒烟：写一帧 JPEG 与一条时序，检查 /dev/shm/newvision_* 落盘。
#include "l6_telemetry/web_sink.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool readFile(const char* path, std::string& out)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return true;
}

}  // namespace

int main()
{
  L6Telemetry::WebSink sink(8);

  // 1) 帧：小图即可，只验证长度前缀 + SOI。
  cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(0, 255, 0));
  sink.writeFrame(frame);
  sink.flush();

  const auto frame_stats = sink.stats();
  if (frame_stats.submitted_frames != 1 || frame_stats.encoded_frames != 1) {
    std::cerr << "web_sink_smoke: FAIL async frame counters\n";
    return 14;
  }

  std::string frame_bytes;
  if (!readFile("/dev/shm/newvision_frame", frame_bytes)) {
    std::cerr << "web_sink_smoke: FAIL cannot read /dev/shm/newvision_frame\n";
    return 1;
  }
  if (frame_bytes.size() < 6) {
    std::cerr << "web_sink_smoke: FAIL frame file too small: " << frame_bytes.size() << "\n";
    return 2;
  }
  std::uint32_t size = 0;
  std::memcpy(&size, frame_bytes.data(), 4);
  if (size + 4 > frame_bytes.size() || size == 0) {
    std::cerr << "web_sink_smoke: FAIL bad jpeg size " << size << "\n";
    return 3;
  }
  const auto* jpeg = reinterpret_cast<const unsigned char*>(frame_bytes.data()) + 4;
  if (jpeg[0] != 0xFF || jpeg[1] != 0xD8) {
    std::cerr << "web_sink_smoke: FAIL missing JPEG SOI\n";
    return 4;
  }

  // 大图由后台等比例缩到 960x720，避免网页调试浪费 CPU 和无线带宽。
  cv::Mat large_frame(1080, 1440, CV_8UC3, cv::Scalar(10, 20, 30));
  sink.writeFrame(large_frame);
  sink.flush();
  if (!readFile("/dev/shm/newvision_frame", frame_bytes)) {
    std::cerr << "web_sink_smoke: FAIL cannot reread resized frame\n";
    return 15;
  }
  std::memcpy(&size, frame_bytes.data(), 4);
  if (size == 0 || size + 4 > frame_bytes.size()) {
    std::cerr << "web_sink_smoke: FAIL bad resized jpeg size\n";
    return 16;
  }
  const std::vector<uchar> encoded(
    frame_bytes.begin() + 4, frame_bytes.begin() + 4 + size);
  const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (decoded.cols != 960 || decoded.rows != 720) {
    std::cerr << "web_sink_smoke: FAIL resized frame "
              << decoded.cols << 'x' << decoded.rows << "\n";
    return 17;
  }

  // 2) 时序：嵌套 telemetryFrame 风格 → 拍平键 + time 数组。
  nlohmann::json sample;
  sample["t"] = 1.25;
  sample["gimbal"]["yaw"] = 10.5;
  sample["gimbal"]["pitch"] = -2.0;
  sample["aim"]["yaw"] = 11.0;
  sample["ekf"]["nis"] = 0.4;
  auto first_sample = sample;
  first_sample.erase("aim");
  sink.writeSeries(first_sample);
  sink.writeSeries(sample);
  sink.flush();

  std::string series_text;
  if (!readFile("/dev/shm/newvision_data.json", series_text)) {
    std::cerr << "web_sink_smoke: FAIL cannot read series json\n";
    return 5;
  }
  const auto series = nlohmann::json::parse(series_text);
  if (!series.contains("time") || series["time"].size() != 2) {
    std::cerr << "web_sink_smoke: FAIL time series length\n";
    return 6;
  }
  if (!series.contains("gimbal/yaw") || series["gimbal/yaw"].size() != 2) {
    std::cerr << "web_sink_smoke: FAIL flattened gimbal/yaw missing\n";
    return 7;
  }
  if (series["gimbal/yaw"][0].get<double>() != 10.5) {
    std::cerr << "web_sink_smoke: FAIL gimbal/yaw value\n";
    return 8;
  }
  if (!series.contains("aim/yaw") || series["aim/yaw"].size() != 2 ||
      series["aim/yaw"][0].get<double>() != 0.0 ||
      series["aim/yaw"][1].get<double>() != 11.0) {
    std::cerr << "web_sink_smoke: FAIL dynamic series key\n";
    return 13;
  }

  // 3) 日志。
  sink.writeLog("boot", nlohmann::json{{"ok", true}});
  sink.writeEvent("mode_change", std::string{"AutoAim"});
  std::string log_text;
  if (!readFile("/dev/shm/newvision_log.json", log_text)) {
    std::cerr << "web_sink_smoke: FAIL log json\n";
    return 9;
  }
  const auto log = nlohmann::json::parse(log_text);
  if (log["boot"]["ok"] != true) {
    std::cerr << "web_sink_smoke: FAIL log boot\n";
    return 10;
  }
  if (!log.contains("events") || log["events"].empty() ||
      log["events"].back()["type"] != "mode_change") {
    std::cerr << "web_sink_smoke: FAIL writeEvent\n";
    return 11;
  }

  std::printf("web_sink_smoke: ok frame_bytes=%zu series_keys=%zu\n",
              frame_bytes.size(), series.size());

  // 4) 录制目录：series.jsonl + frames/*.jpg。
  {
    const std::string record_dir = "/tmp/newvision_web_record";
    std::filesystem::remove_all(record_dir);
    L6Telemetry::WebSink recorder(8, record_dir);
    recorder.writeFrame(frame);
    recorder.writeSeries(sample);
    recorder.flush();

    std::string jsonl;
    if (!readFile((record_dir + "/series.jsonl").c_str(), jsonl) ||
        jsonl.find("gimbal") == std::string::npos) {
      std::cerr << "web_sink_smoke: FAIL record series.jsonl\n";
      return 10;
    }
    if (!std::filesystem::exists(record_dir + "/frames/frame_000000.jpg")) {
      std::cerr << "web_sink_smoke: FAIL record frame missing\n";
      return 11;
    }
    std::printf("web_sink_smoke: ok record_dir\n");
  }

  return 0;
}
