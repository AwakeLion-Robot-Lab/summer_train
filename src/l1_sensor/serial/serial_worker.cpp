#include "l1_sensor/serial/serial_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

namespace L1Sensor {

// 创建串口端口和协议解析器，并应用丢包检测开关配置。
SerialWorker::SerialWorker(SerialConfig config)
    : config_(std::move(config)), port_(config_) {
  protocol_.setPacketLossCheckEnable(config_.packet_loss_check_enable);
}

// 析构时停止收发线程，确保线程退出后再释放串口。
SerialWorker::~SerialWorker() { stop(); }

// 启动串口收发线程；串口暂时打不开时，线程内部会持续重连。
bool SerialWorker::start() {
  if (!config_.enable) {
    L6Telemetry::logInfo("serial worker disabled");
    return false;
  }

  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);

  // 当前值与期望值相等时改为 true，用来避免重复启动线程。
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return true;
  }

  try {
    rx_thread_ = std::thread(&SerialWorker::rxLoop, this);
    tx_thread_ = std::thread(&SerialWorker::txLoop, this);
  } catch (const std::exception &e) {
    running_ = false;
    if (rx_thread_.joinable()) {
      rx_thread_.join();
    }
    port_.close();
    L6Telemetry::logError("serial worker start failed", e.what());
    return false;
  }

  L6Telemetry::logInfo("serial worker started", config_.device);
  return true;
}

// 请求线程停止，等待收发线程退出，并关闭串口。
void SerialWorker::stop() {
  std::lock_guard<std::mutex> lifecycle_lock(lifecycle_mutex_);

  if (!running_.exchange(false)) {
    return;
  }

  if (rx_thread_.joinable()) {
    rx_thread_.join();
  }

  if (tx_thread_.joinable()) {
    tx_thread_.join();
  }

  port_.close();
}

// 返回当前串口 worker 是否处于运行状态。
bool SerialWorker::isRunning() const { return running_.load(); }

// 更新发送线程下一次要发给下位机的控制命令。
void SerialWorker::updateCommand(const L5Control::SerialCommand &command) {
  std::lock_guard<std::mutex> lock(command_mutex_);
  latest_command_ = command;
  last_command_update_ = std::chrono::steady_clock::now();
  has_command_ = true;
  command_timed_out_ = false;
}

// 获取最近一次从串口解析出的机器人状态。
std::optional<RobotState> SerialWorker::latestState() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return latest_state_;
}

// 运行时开关 seq 丢包检测。
void SerialWorker::setPacketLossCheckEnable(bool enable) {
  protocol_.setPacketLossCheckEnable(enable);
}

// 查询当前是否启用 seq 丢包检测。
bool SerialWorker::packetLossCheckEnable() const {
  return protocol_.packetLossCheckEnable();
}

// 获取协议层累计检测到的丢包数量。
std::uint64_t SerialWorker::droppedPacketCount() const {
  return protocol_.droppedPacketCount();
}

// 返回成功完整写入底层串口的控制帧数量；不代表下位机已经执行该命令。
std::uint64_t SerialWorker::sentCommandCount() const {
  return sent_command_count_.load();
}

// 获取最近一次确认完整写入串口的命令，用于运行时诊断和硬件联调。
std::optional<L5Control::SerialCommand>
SerialWorker::latestSentCommand() const {
  std::lock_guard<std::mutex> lock(sent_command_mutex_);
  return latest_sent_command_;
}

// 返回通过 CRC 和协议长度检查后解析出的下位机状态帧数量。
std::uint64_t SerialWorker::receivedStateCount() const {
  return received_state_count_.load();
}

// 返回底层串口未完整写入的控制帧数量。
std::uint64_t SerialWorker::failedCommandCount() const {
  return failed_command_count_.load();
}

// 确保串口打开；如果串口断开，会串行化重连尝试。
bool SerialWorker::ensureOpen() {
  if (port_.isOpen()) {
    return true;
  }

  return port_.open();
}

// 接收线程：读取串口字节、解包 RobotState，并维护云台 RPY 历史。
void SerialWorker::rxLoop() {
  const auto reconnect_interval =
      std::chrono::milliseconds(std::max(config_.reconnect_interval_ms, 1));
  const auto rx_buffer_size = std::max<std::size_t>(config_.rx_buffer_size, 1);
  std::vector<std::uint8_t> buffer(rx_buffer_size);

  while (running_.load()) {
    if (!ensureOpen()) {
      std::this_thread::sleep_for(reconnect_interval);
      continue;
    }

    const auto bytes_read = port_.read(buffer);
    if (bytes_read == 0) {
      continue;
    }

    const auto states = protocol_.feed(
        std::span<const std::uint8_t>{buffer.data(), bytes_read});
    if (states.empty()) {
      continue;
    }

    received_state_count_.fetch_add(states.size());

    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto &state : states) {
      if (!gimbal_history_.empty() &&
          state.timestamp <= gimbal_history_.back().timestamp) {
        L6Telemetry::logWarn(
            "serial gimbal history timestamp is not increasing");
        continue;
      }

      latest_state_ = state;
      gimbal_history_.push_back(state);

      while (gimbal_history_.size() > max_gimbal_history_size_) {
        gimbal_history_.pop_front();
      }
    }
  }
}

// 发送线程：按配置频率把最新控制命令编码后写入串口。
void SerialWorker::txLoop() {
  const int tx_rate_hz = std::max(config_.tx_rate_hz, 1);
  const auto period =
      std::chrono::microseconds(std::max(1, 1'000'000 / tx_rate_hz));
  const auto command_timeout =
      std::chrono::milliseconds(std::max(config_.command_timeout_ms, 1));
  const auto reconnect_interval =
      std::chrono::milliseconds(std::max(config_.reconnect_interval_ms, 1));

  while (running_.load()) {
    const auto next_tick = std::chrono::steady_clock::now() + period;

    if (!ensureOpen()) {
      std::this_thread::sleep_for(reconnect_interval);
      continue;
    }

    L5Control::SerialCommand command{};
    bool command_expired = false;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      const auto now = std::chrono::steady_clock::now();
      const bool command_is_fresh =
          has_command_ && now - last_command_update_ <= command_timeout;
      if (command_is_fresh) {
        command = latest_command_;
        command_timed_out_ = false;
      } else {
        // 协议没有独立的 hold/valid
        // 位；保留最后设定值以避免云台突跳，强制关火。
        if (has_command_) {
          command = latest_command_;
          if (!command_timed_out_) {
            command_timed_out_ = true;
            command_expired = true;
          }
        }
        command.shoot = false;
      }
    }

    if (command_expired) {
      L6Telemetry::logWarn("serial command expired; sending safe command",
                           command_timeout.count());
    }

    const auto bytes = protocol_.encodeCommand(command);
    if (port_.write(bytes) == bytes.size()) {
      {
        std::lock_guard<std::mutex> lock(sent_command_mutex_);
        latest_sent_command_ = command;
      }
      sent_command_count_.fetch_add(1);
    } else {
      failed_command_count_.fetch_add(1);
    }

    std::this_thread::sleep_until(next_tick);
  }
}

// 按时间戳查询云台姿态；找到前后两帧 RPY 后再转四元数并 slerp。
std::optional<Eigen::Quaterniond> SerialWorker::gimbalPoseAt(
    std::chrono::steady_clock::time_point timestamp) const {
  std::lock_guard<std::mutex> lock(state_mutex_);

  if (gimbal_history_.empty()) {
    L6Telemetry::logWarn("gimbal pose history empty");
    return std::nullopt;
  }

  const auto to_quaternion = [](const RobotState &state) {
    return L6Telemetry::rpyToQuaternion(state.rpy.roll, state.rpy.pitch,
                                        state.rpy.yaw);
  };

  if (timestamp <= gimbal_history_.front().timestamp) {
    L6Telemetry::logDebug("gimbal pose before history");
    return to_quaternion(gimbal_history_.front());
  }

  if (timestamp >= gimbal_history_.back().timestamp) {
    L6Telemetry::logDebug("gimbal pose after history");
    return to_quaternion(gimbal_history_.back());
  }

  const auto upper = std::lower_bound(
      gimbal_history_.begin(), gimbal_history_.end(), timestamp,
      [](const RobotState &state,
         const std::chrono::steady_clock::time_point &t) {
        return state.timestamp < t;
      });

  if (upper == gimbal_history_.begin() || upper == gimbal_history_.end()) {
    L6Telemetry::logWarn("gimbal pose search invalid");
    return std::nullopt;
  }

  const auto &before = *(upper - 1);
  const auto &after = *upper;

  const double interval =
      std::chrono::duration<double>(after.timestamp - before.timestamp).count();
  if (interval <= 0.0) {
    L6Telemetry::logWarn("gimbal pose interval invalid");
    return to_quaternion(before);
  }

  const double elapsed =
      std::chrono::duration<double>(timestamp - before.timestamp).count();
  const double ratio = elapsed / interval;

  return L6Telemetry::slerpQuaternion(to_quaternion(before),
                                      to_quaternion(after), ratio);
}

} // namespace L1Sensor
