#include "l1_sensor/serial/serial_config.hpp"

#include <algorithm>
#include <cmath>
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

// 读取 3x3 旋转矩阵；缺失、形状错误或不是合法旋转时保留默认值。
// 这里不接受"接近正交"以外的输入，避免把一个缩放或镜像矩阵当成姿态外参。
Eigen::Matrix3d readRotation(const YAML::Node &node, const std::string &key,
                             const Eigen::Matrix3d &default_value) {
  if (!node[key]) {
    return default_value;
  }

  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  try {
    const YAML::Node &rows = node[key];
    if (!rows.IsSequence() || rows.size() != 3) {
      L6Telemetry::logWarn("serial config rotation must have 3 rows", key);
      return default_value;
    }

    for (std::size_t row = 0; row < 3; ++row) {
      if (!rows[row].IsSequence() || rows[row].size() != 3) {
        L6Telemetry::logWarn("serial config rotation row must have 3 columns",
                             key);
        return default_value;
      }
      for (std::size_t col = 0; col < 3; ++col) {
        rotation(static_cast<int>(row), static_cast<int>(col)) =
            rows[row][col].as<double>();
      }
    }
  } catch (const YAML::Exception &e) {
    L6Telemetry::logWarn("serial config invalid rotation", key, e.what());
    return default_value;
  }

  constexpr double kRotationTolerance = 1e-3;
  const bool orthonormal =
      rotation.allFinite() &&
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <=
          kRotationTolerance &&
      std::abs(rotation.determinant() - 1.0) <= kRotationTolerance;
  if (!orthonormal) {
    L6Telemetry::logWarn("serial config rotation is not a valid rotation", key);
    return default_value;
  }

  return rotation;
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
  config.R_imu_barrel =
      readRotation(yaml, "R_imu_barrel", config.R_imu_barrel);

  // 这条外参一旦配错，整个世界系姿态都会错，因此启动时明确记录使用的约定。
  if (config.imuBarrelRotationNeeded()) {
    L6Telemetry::logInfo("serial config applies R_imu_barrel conversion");
  } else {
    L6Telemetry::logInfo(
        "serial config assumes MCU reports pose in barrel frame");
  }

  normalize(config);
  return config;
}

} // namespace L1Sensor
