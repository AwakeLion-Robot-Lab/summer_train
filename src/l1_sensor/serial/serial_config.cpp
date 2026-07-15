#include "l1_sensor/serial/serial_config.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include "l6_telemetry/logger.hpp"
#include "yaml.hpp"

namespace L1Sensor {
namespace {

// 从 YAML 读取可选字段；字段不存在时保留默认值。
template <typename T>
T readOptional(const YAML::Node &node, const std::string &key,
               const T &default_value) {
  if (!node[key]) {
    return default_value;
  }

  try {
    return node[key].as<T>();
  } catch (const YAML::Exception &e) {
    L6Telemetry::logWarn("serial config invalid field", key, e.what());
    return default_value;
  }
}

// 修正明显非法的串口参数，避免运行时除零或空设备名。
void normalize(SerialConfig &config) {
  if (config.device.empty()) {
    L6Telemetry::logWarn("serial config device empty, use default");
    config.device = "/dev/ttyACM0";
  }

  if (config.baud_rate <= 0) {
    L6Telemetry::logWarn("serial config baud rate invalid, use default");
    config.baud_rate = 1000000;
  }

  config.read_timeout_ms = std::max(config.read_timeout_ms, 0);
  config.tx_rate_hz = std::max(config.tx_rate_hz, 1);
  config.command_timeout_ms = std::max(config.command_timeout_ms, 1);
  config.reconnect_interval_ms = std::max(config.reconnect_interval_ms, 1);
  config.rx_buffer_size = std::max<std::size_t>(config.rx_buffer_size, 1);
}

} // namespace

// 从 YAML 文件读取串口配置，并对非法值做最小修正。
SerialConfig loadSerialConfig(const std::string &config_path) {
  SerialConfig config;
  const auto yaml = tools::load(config_path);

  config.enable = readOptional(yaml, "enable", config.enable);
  config.device = readOptional(yaml, "device", config.device);
  config.baud_rate = readOptional(yaml, "baud_rate", config.baud_rate);
  config.read_timeout_ms =
      readOptional(yaml, "read_timeout_ms", config.read_timeout_ms);
  config.tx_rate_hz = readOptional(yaml, "tx_rate_hz", config.tx_rate_hz);
  config.command_timeout_ms =
      readOptional(yaml, "command_timeout_ms", config.command_timeout_ms);
  config.reconnect_interval_ms =
      readOptional(yaml, "reconnect_interval_ms", config.reconnect_interval_ms);
  config.rx_buffer_size =
      readOptional(yaml, "rx_buffer_size", config.rx_buffer_size);
  config.packet_loss_check_enable = readOptional(
      yaml, "packet_loss_check_enable", config.packet_loss_check_enable);

  normalize(config);
  return config;
}

} // namespace L1Sensor
