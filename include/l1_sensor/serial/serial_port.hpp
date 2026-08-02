#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <span>

#include "l1_sensor/serial/serial_config.hpp"

namespace serial {
class Serial;
}

namespace L1Sensor {

class SerialPort {
public:
  explicit SerialPort(SerialConfig config);
  ~SerialPort();

  SerialPort(const SerialPort&) = delete;
  SerialPort& operator=(const SerialPort&) = delete;

  // 打开底层串口设备，成功返回 true。
  bool open();

  // 关闭串口设备，释放底层文件句柄。
  void close();

  // 查询当前串口是否已经打开。
  bool isOpen() const;

  // 从串口读取原始字节，返回实际读到的字节数。
  std::size_t read(std::span<std::uint8_t> buffer);

  // 向串口写入原始字节，返回实际写出的字节数。
  std::size_t write(std::span<const std::uint8_t> data);

private:
  bool isOpenLocked() const;
  void closeLocked();

  SerialConfig config_;
  std::unique_ptr<serial::Serial> serial_;
  mutable std::shared_mutex lifecycle_mutex_;
  bool open_failure_logged_ = false;
};

}  // namespace L1Sensor
