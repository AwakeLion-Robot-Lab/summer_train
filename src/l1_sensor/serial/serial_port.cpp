#include "l1_sensor/serial/serial_port.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <mutex>
#include <utility>

#include "l6_telemetry/logger.hpp"
#include "serial/serial.h"

namespace L1Sensor {

// 根据配置创建底层 serial::Serial 对象，并设置读超时时间。
SerialPort::SerialPort(SerialConfig config) : config_(std::move(config)) {
  const auto read_timeout_ms =
      static_cast<std::uint32_t>(std::max(config_.read_timeout_ms, 0));
  auto timeout = serial::Timeout::simpleTimeout(read_timeout_ms);
  serial_ = std::make_unique<serial::Serial>();
  serial_->setPort(config_.device);
  serial_->setBaudrate(static_cast<std::uint32_t>(config_.baud_rate));
  serial_->setBytesize(serial::eightbits);
  serial_->setParity(serial::parity_none);
  serial_->setStopbits(serial::stopbits_one);
  serial_->setFlowcontrol(serial::flowcontrol_none);
  serial_->setTimeout(timeout);
  L6Telemetry::logInfo("serial configured", config_.device, config_.baud_rate);
}

// 析构时自动关闭串口，避免文件句柄泄漏。
SerialPort::~SerialPort() { close(); }

// 打开串口设备，异常时返回 false。
bool SerialPort::open() {
  std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);

  try {
    if (isOpenLocked()) {
      return true;
    }

    serial_->open();
    open_failure_logged_ = false;
    L6Telemetry::logInfo("serial opened", config_.device);
    return isOpenLocked();
  } catch (const std::exception &e) {
    closeLocked();
    if (!open_failure_logged_) {
      L6Telemetry::logWarn("serial open failed", config_.device, e.what());
      open_failure_logged_ = true;
    }
    return false;
  }
}

// 关闭串口设备，关闭失败时吞掉异常。
void SerialPort::close() {
  std::unique_lock<std::shared_mutex> lock(lifecycle_mutex_);
  closeLocked();
}

// 调用方持有 lifecycle_mutex_ 的独占锁；避免 close 与 read/write 并发访问 SDK
// 句柄。
void SerialPort::closeLocked() {
  try {
    if (isOpenLocked()) {
      serial_->close();
      L6Telemetry::logInfo("serial closed", config_.device);
    }
  } catch (const std::exception &e) {
    L6Telemetry::logWarn("serial close failed", config_.device, e.what());
  }
}

// 查询底层串口对象是否存在且处于打开状态。
bool SerialPort::isOpen() const {
  std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
  return isOpenLocked();
}

// 调用方已持有 lifecycle_mutex_；serial_ 在构造后不再替换。
bool SerialPort::isOpenLocked() const { return serial_ && serial_->isOpen(); }

// 读取一段原始字节，串口未打开、buffer 为空或异常时返回 0。
std::size_t SerialPort::read(std::span<std::uint8_t> buffer) {
  bool close_after_error = false;
  {
    std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
    if (!isOpenLocked() || buffer.empty()) {
      return 0;
    }

    try {
      return serial_->read(buffer.data(), buffer.size());
    } catch (const std::exception &e) {
      L6Telemetry::logWarn("serial read failed", config_.device, e.what());
      close_after_error = true;
    }
  }

  if (close_after_error) {
    close();
  }
  return 0;
}

// 写入一段原始字节，串口未打开、数据为空或异常时返回 0。
std::size_t SerialPort::write(std::span<const std::uint8_t> data) {
  bool close_after_error = false;
  {
    std::shared_lock<std::shared_mutex> lock(lifecycle_mutex_);
    if (!isOpenLocked() || data.empty()) {
      return 0;
    }

    try {
      const auto bytes_written = serial_->write(data.data(), data.size());
      if (bytes_written != data.size()) {
        L6Telemetry::logWarn("serial write incomplete", bytes_written,
                             data.size());
      }
      return bytes_written;
    } catch (const std::exception &e) {
      L6Telemetry::logWarn("serial write failed", config_.device, e.what());
      close_after_error = true;
    }
  }

  if (close_after_error) {
    close();
  }
  return 0;
}

} // namespace L1Sensor
