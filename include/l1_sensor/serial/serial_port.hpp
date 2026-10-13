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

/**
 * @brief 串口设备的线程安全封装，只收发原始字节
 */
class SerialPort {
public:
  /**
   * @brief 保存配置，不立即打开设备
   * @param config 串口配置（设备路径、波特率、读超时）
   */
  explicit SerialPort(SerialConfig config);

  /**
   * @brief 析构时关闭设备
   */
  ~SerialPort();

  SerialPort(const SerialPort&) = delete;
  SerialPort& operator=(const SerialPort&) = delete;

  /**
   * @brief 打开底层串口设备
   * @return 成功返回 true
   */
  bool open();

  /**
   * @brief 关闭串口设备，释放底层文件句柄
   */
  void close();

  /**
   * @brief 查询当前串口是否已经打开
   * @return 已打开返回 true
   */
  bool isOpen() const;

  /**
   * @brief 从串口读取原始字节
   * @param buffer 接收缓冲区，最多读满它
   * @return 实际读到的字节数
   */
  std::size_t read(std::span<std::uint8_t> buffer);

  /**
   * @brief 向串口写入原始字节
   * @param data 待写出的数据
   * @return 实际写出的字节数
   */
  std::size_t write(std::span<const std::uint8_t> data);

private:
  /**
   * @brief isOpen() 的无锁版本，调用方须已持有 lifecycle_mutex_
   * @return 已打开返回 true
   */
  bool isOpenLocked() const;

  /**
   * @brief close() 的无锁版本，调用方须已持有 lifecycle_mutex_
   */
  void closeLocked();

  /// 串口配置
  SerialConfig config_;

  /// 底层串口对象，未打开时为空
  std::unique_ptr<serial::Serial> serial_;

  /// 保护 serial_ 的生命周期：读写共享，开关独占
  mutable std::shared_mutex lifecycle_mutex_;

  /// 打开失败的日志只打一次，避免重连时刷屏
  bool open_failure_logged_ = false;
};

}  // namespace L1Sensor
