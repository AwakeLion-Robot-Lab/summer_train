#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/robot_state.hpp"
#include "l1_sensor/talos/shm_layout.hpp"
#include "l5_control/serial_command.hpp"

#include <Eigen/Geometry>
#include <chrono>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <string>

namespace runtime {

struct RuntimeIoStats {
  std::uint64_t received_states{};
  std::uint64_t dropped_packets{};
  std::uint64_t skipped_bytes{};
  std::uint64_t sent_commands{};
  std::uint64_t failed_commands{};
  std::uint64_t pose_before_history{};
  std::uint64_t pose_after_history{};
  std::uint64_t camera_captured_frames{};
  std::uint64_t camera_dropped_frames{};
  std::uint64_t algorithm_received_frames{};
};

// 算法循环只依赖这一组语义。硬件串口和 Talos 共享内存各自处理传输细节，
// 避免为了仿真复制一份 L2-L5 或在算法层插入坐标轴特判。
class RuntimeIo {
public:
  virtual ~RuntimeIo() = default;
  virtual bool start() = 0;
  virtual bool read(cv::Mat& image, std::chrono::steady_clock::time_point& timestamp) = 0;
  virtual const std::optional<L1Sensor::CameraCalibration>& calibration() const noexcept = 0;
  virtual std::optional<L1Sensor::RobotState> latestState() const = 0;
  virtual bool waitPose(std::chrono::steady_clock::time_point timestamp) const = 0;
  virtual std::optional<Eigen::Quaterniond> gimbalPoseAt(
    std::chrono::steady_clock::time_point timestamp) const = 0;
  virtual std::optional<Eigen::Quaterniond> latestGimbalPose() const = 0;
  virtual void updateCommand(const L5Control::SerialCommand& command) = 0;
  virtual RuntimeIoStats stats() const = 0;
  virtual std::optional<L1Sensor::talos::GroundTruthBatch> groundTruth() const = 0;
  virtual std::string name() const = 0;
  virtual void stop() = 0;
  // 在线调参：透传到相机驱动。Talos/仿真返回 false。
  virtual bool setExposureGain(double /*exposure_ms*/, double /*gain*/) { return false; }
};

// camera_name: talos 选择仿真后端，其余配置保持现有硬件行为。
std::shared_ptr<RuntimeIo> makeRuntimeIo(const std::string& camera_config_path);

} // namespace runtime
