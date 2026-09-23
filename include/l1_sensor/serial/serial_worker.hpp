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

  // 返回累计被 SOF 搜索丢弃的字节数，含义见 SerialProtocol::skippedByteCount()。
  std::uint64_t skippedByteCount() const;

  // 返回本 worker 生命周期内成功完整写入串口的控制帧数量。
  std::uint64_t sentCommandCount() const;

  // 返回最近一次完整写入串口的命令；未成功发送过时返回空。
  std::optional<L5Control::SerialCommand> latestSentCommand() const;

  // 返回本 worker 生命周期内成功解析出的下位机状态帧数量。
  std::uint64_t receivedStateCount() const;

  // 返回写入不完整或发生写入错误的控制帧数量。
  std::uint64_t failedCommandCount() const;

  // 根据图像时间戳查询云台姿态；内部会在历史 RPY 中找前后两帧并 slerp。
  // 返回的姿态已经是 barrel -> world，可直接交给 L3，无需再补轴向转换。
  //
  // 时间戳落在历史区间外时会夹到边界并**只累加计数器**，不打日志：这一路每帧
  // 都会走到，逐帧打印会把真正的告警冲掉。频次用 poseBeforeHistoryCount() /
  // poseAfterHistoryCount() 看。
  std::optional<Eigen::Quaterniond> gimbalPoseAt(
    std::chrono::steady_clock::time_point timestamp) const;

  // 最新一次云台姿态，不做插值。"现在"之后不可能有采样，所以想要当前姿态时
  // 用这个而不是 gimbalPoseAt(now())——后者会无谓地走一遍越界分支并计数，
  // 把统计污染成"每帧都越界"，真正该关注的图像时刻越界反而看不出来。
  std::optional<Eigen::Quaterniond> latestGimbalPose() const;

  // gimbalPoseAt() 因时间戳越界而退化为边界姿态的累计次数。分早于/晚于历史
  // 两种：前者说明图像比姿态历史还老（丢帧或历史太短），后者说明图像时间戳
  // 比最新姿态还新（相机时间戳未补曝光与传输耗时，或 rx 停顿）。
  std::uint64_t poseBeforeHistoryCount() const;
  std::uint64_t poseAfterHistoryCount() const;

private:
  // 把下位机 IMU 约定下的姿态按 config_.R_imu_barrel 转换到 barrel 约定。
  Eigen::Quaterniond toBarrelPose(
    const Eigen::Quaterniond& q_world_imu) const;

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
  mutable std::atomic<std::uint64_t> pose_before_history_count_{0};
  mutable std::atomic<std::uint64_t> pose_after_history_count_{0};

  mutable std::mutex command_mutex_;
  L5Control::SerialCommand latest_command_;
  std::chrono::steady_clock::time_point last_command_update_{};
  bool has_command_ = false;
  bool command_timed_out_ = true;

  mutable std::mutex sent_command_mutex_;
  std::optional<L5Control::SerialCommand> latest_sent_command_;
};

}  // namespace L1Sensor
