#pragma once

#include <Eigen/Geometry>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

#include "l1_sensor/serial/robot_state.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_port.hpp"
#include "l1_sensor/serial/serial_protocol.hpp"
#include "l5_control/serial_command.hpp"

namespace L1Sensor {

class SerialWorker {
public:
  explicit SerialWorker(SerialConfig config);
  ~SerialWorker();

  SerialWorker(const SerialWorker&) = delete;
  SerialWorker& operator=(const SerialWorker&) = delete;

  // 启动串口收发线程。
  bool start();

  // 停止串口收发线程并关闭串口。
  void stop();

  // 查询串口线程是否正在运行。
  bool isRunning() const;

  // 更新待发送给下位机的最新控制命令，并续期其有效时间。
  void updateCommand(const L5Control::SerialCommand& command);

  // 获取最近一次成功解析出的机器人状态。
  std::optional<RobotState> latestState() const;

  // 开关 seq 丢包检测。
  void setPacketLossCheckEnable(bool enable);

  // 查询当前是否启用 seq 丢包检测。
  bool packetLossCheckEnable() const;

  // 返回累计检测到的丢包数量。
  std::uint64_t droppedPacketCount() const;

  // 返回本 worker 生命周期内成功完整写入串口的控制帧数量。
  std::uint64_t sentCommandCount() const;

  // 返回最近一次完整写入串口的命令；未成功发送过时返回空。
  std::optional<L5Control::SerialCommand> latestSentCommand() const;

  // 返回本 worker 生命周期内成功解析出的下位机状态帧数量。
  std::uint64_t receivedStateCount() const;

  // 返回写入不完整或发生写入错误的控制帧数量。
  std::uint64_t failedCommandCount() const;

  // 根据图像时间戳查询云台姿态；内部会在历史 RPY 中找前后两帧并 slerp。
  std::optional<Eigen::Quaterniond> gimbalPoseAt(
    std::chrono::steady_clock::time_point timestamp) const;

private:
  // 确保串口处于打开状态；断开后会尝试重新打开。
  bool ensureOpen();

  // 串口接收线程：读字节、解协议、更新状态和云台历史。
  void rxLoop();

  // 串口发送线程：按固定频率发送最新控制命令。
  void txLoop();

  SerialConfig config_;
  SerialPort port_;
  SerialProtocol protocol_;

  std::atomic<bool> running_{false};
  std::thread rx_thread_;
  std::thread tx_thread_;
  std::mutex lifecycle_mutex_;

  mutable std::mutex state_mutex_;
  std::optional<RobotState> latest_state_;
  std::deque<RobotState> gimbal_history_;
  std::size_t max_gimbal_history_size_ = 64;
  std::atomic<std::uint64_t> sent_command_count_{0};
  std::atomic<std::uint64_t> received_state_count_{0};
  std::atomic<std::uint64_t> failed_command_count_{0};

  mutable std::mutex command_mutex_;
  L5Control::SerialCommand latest_command_;
  std::chrono::steady_clock::time_point last_command_update_{};
  bool has_command_ = false;
  bool command_timed_out_ = true;

  mutable std::mutex sent_command_mutex_;
  std::optional<L5Control::SerialCommand> latest_sent_command_;
};

}  // namespace L1Sensor
