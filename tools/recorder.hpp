#pragma once

#include <Eigen/Geometry>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

namespace tools {

struct RecorderConfig {
  bool enabled = false;
  double fps = 30.0;
  std::filesystem::path output_dir = "records";
};

// 从 YAML 顶层的 recorder 节点读取配置；节点或字段缺失时保留默认值。
RecorderConfig loadRecorderConfig(const std::string& config_path);

class Recorder {
public:
  explicit Recorder(RecorderConfig config);
  ~Recorder();

  Recorder(const Recorder&) = delete;
  Recorder& operator=(const Recorder&) = delete;
  Recorder(Recorder&&) = delete;
  Recorder& operator=(Recorder&&) = delete;

  // 成功把一组图像、四元数和时间戳加入写盘队列时返回 true。
  bool record(
    const cv::Mat& image,
    const Eigen::Quaterniond& quaternion,
    std::chrono::steady_clock::time_point timestamp);

  // 停止接收新样本，排空队列并关闭输出文件；可以重复调用。
  void stop();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tools
