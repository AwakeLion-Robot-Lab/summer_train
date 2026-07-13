#pragma once

#include <cstddef>
#include <string>

namespace L1Sensor {

// 串口运行参数；默认值用于快速跑通，也可由 YAML 覆盖。
struct SerialConfig {
  bool enable = true;
  std::string device = "/dev/ttyACM0";
  int baud_rate = 1000000;
  int read_timeout_ms = 20;
  int tx_rate_hz = 200;
  int command_timeout_ms = 100;
  int reconnect_interval_ms = 500;
  std::size_t rx_buffer_size = 256;
  bool packet_loss_check_enable = true;
};

// 从 YAML 文件读取串口配置，缺失字段使用 SerialConfig 默认值。
SerialConfig loadSerialConfig(const std::string& config_path);

}  // namespace L1Sensor
