#pragma once

#include <Eigen/Geometry>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "l1_sensor/serial/robot_state.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_port.hpp"
#include "l1_sensor/serial/serial_protocol.hpp"
#include "l5_control/serial_command.hpp"

namespace L1Sensor {

/**
 * @brief 串口收发线程
 *
 * 接收线程解析下位机状态并记录云台姿态历史，发送线程按固定频率下发最新控制
 * 命令。串口断开后自动重连，并统计丢包、发送和失败帧数。
 */
class SerialWorker {
public:
  /**
   * @brief 保存配置，不启动线程
   * @param config 串口配置
   * @param receive_only 为 true 时只启动接收线程，不发送任何控制命令
   */
  explicit SerialWorker(SerialConfig config, bool receive_only = false);

  /**
   * @brief 析构时停止收发线程，确保线程退出后再释放串口
   */
  ~SerialWorker();

  SerialWorker(const SerialWorker&) = delete;
  SerialWorker& operator=(const SerialWorker&) = delete;

  /**
   * @brief 启动串口收发线程
   *
   * 串口暂时打不开时，线程内部会持续重连。
   *
   * @return config.enable 为 false 或线程创建失败时返回 false；已在运行时返回 true
   */
  bool start();

  /**
   * @brief 停止串口收发线程并关闭串口
   */
  void stop();

  /**
   * @brief 查询串口线程是否正在运行
   * @return 正在运行返回 true
   */
  bool isRunning() const;

  /**
   * @brief 更新待发送给下位机的最新控制命令，并续期其有效时间
   * @param command 控制命令，yaw/pitch 为世界系绝对角
   */
  void updateCommand(const L5Control::SerialCommand& command);

  /**
   * @brief 获取最近一次成功解析出的机器人状态
   * @return 机器人状态；还没收到过时为空
   */
  std::optional<RobotState> latestState() const;

  /**
   * @brief 开关 seq 丢包检测
   * @param enable true 为开启
   */
  void setPacketLossCheckEnable(bool enable);

  /**
   * @brief 查询当前是否启用 seq 丢包检测
   * @return 启用返回 true
   */
  bool packetLossCheckEnable() const;

  /**
   * @brief 累计检测到的丢包数量
   * @return 丢包数
   */
  std::uint64_t droppedPacketCount() const;

  /**
   * @brief 本 worker 生命周期内成功完整写入串口的控制帧数量
   * @return 发送成功的帧数
   */
  std::uint64_t sentCommandCount() const;

  /**
   * @brief 最近一次完整写入串口的命令
   * @return 最近发送的命令；未成功发送过时为空
   */
  std::optional<L5Control::SerialCommand> latestSentCommand() const;

  /**
   * @brief 本 worker 生命周期内成功解析出的下位机状态帧数量
   * @return 接收成功的帧数
   */
  std::uint64_t receivedStateCount() const;

  /**
   * @brief 写入不完整或发生写入错误的控制帧数量
   * @return 发送失败的帧数
   */
  std::uint64_t failedCommandCount() const;

  /**
   * @brief 根据图像时间戳查询云台姿态
   *
   * 内部会在历史 RPY 中找前后两帧并 slerp。返回的姿态已经是 barrel -> world，
   * 可直接交给 L3，无需再补轴向转换。
   *
   * @param timestamp 图像曝光时刻，不要传"现在"
   * @return R_world_barrel 对应的四元数；没有姿态历史时为空
   */
  std::optional<Eigen::Quaterniond> gimbalPoseAt(
    std::chrono::steady_clock::time_point timestamp) const;

  /**
   * @brief 阻塞到收到一包时间戳晚于 timestamp 的姿态，或等满 pose_wait_ms
   *
   * 做法同 sp_vision 的 imu_at：图像刚到手时，它之后的那包姿态往往还在路上，
   * 先等到它，gimbalPoseAt 才是前后两包插值而不是拿最新一包顶替。
   *
   * @param timestamp 要覆盖到的姿态查询时刻
   * @return 等到了返回 true，等满超时返回 false
   */
  bool waitPose(std::chrono::steady_clock::time_point timestamp) const;

  /**
   * @brief 姿态历史覆盖的时间范围
   *
   * gimbalPoseAt 查范围之外的时刻会静默取边界值，需要确认是真插值的调用方先查它。
   *
   * @return [最早, 最新]；历史为空时为空
   */
  std::optional<std::pair<std::chrono::steady_clock::time_point,
                          std::chrono::steady_clock::time_point>>
  poseSpan() const;

private:
  /**
   * @brief 把下位机 IMU 约定下的姿态按 config_.R_imu_barrel 转换到 barrel 约定
   * @param q_world_imu IMU 系到世界系的旋转
   * @return R_world_imu * R_imu_barrel
   */
  Eigen::Quaterniond toBarrelPose(
    const Eigen::Quaterniond& q_world_imu) const;

  /**
   * @brief 确保串口处于打开状态，断开后会尝试重新打开
   * @return 串口可用时返回 true
   */
  bool ensureOpen();

  /**
   * @brief 串口接收线程：读字节、解协议、更新状态和云台历史
   */
  void rxLoop();

  /**
   * @brief 串口发送线程：按固定频率发送最新控制命令
   */
  void txLoop();

  /// 串口配置
  SerialConfig config_;

  /// 录制等只读场景不启动发送线程，避免向云台发送默认角度命令。
  bool receive_only_{false};

  /// 串口设备
  SerialPort port_;

  /// 帧协议编解码
  SerialProtocol protocol_;

  /// 收发线程是否在运行
  std::atomic<bool> running_{false};

  /// 接收线程
  std::thread rx_thread_;

  /// 发送线程
  std::thread tx_thread_;

  /// 串行化 start() / stop()
  std::mutex lifecycle_mutex_;

  /// 保护 latest_state_ 与 gimbal_history_
  mutable std::mutex state_mutex_;

  /// 新姿态到达时唤醒 waitPose()
  mutable std::condition_variable pose_cv_;

  /// 最近一次解析出的状态
  std::optional<RobotState> latest_state_;

  /// 云台姿态历史，按时间戳递增，用于插值
  std::deque<RobotState> gimbal_history_;

  /// gimbal_history_ 的最大长度
  std::size_t max_gimbal_history_size_ = 64;

  /// 发送成功的帧数
  std::atomic<std::uint64_t> sent_command_count_{0};

  /// 接收成功的帧数
  std::atomic<std::uint64_t> received_state_count_{0};

  /// 发送失败的帧数
  std::atomic<std::uint64_t> failed_command_count_{0};

  /// 保护下面四个命令相关字段
  mutable std::mutex command_mutex_;

  /// 待发送的最新命令
  L5Control::SerialCommand latest_command_;

  /// latest_command_ 最近一次更新的时刻，用于判断命令是否过期
  std::chrono::steady_clock::time_point last_command_update_{};

  /// 是否收到过命令
  bool has_command_ = false;

  /// 命令已过期；只在过期瞬间打一次日志
  bool command_timed_out_ = true;

  /// 保护 latest_sent_command_
  mutable std::mutex sent_command_mutex_;

  /// 最近一次完整写入串口的命令
  std::optional<L5Control::SerialCommand> latest_sent_command_;
};

}  // namespace L1Sensor
