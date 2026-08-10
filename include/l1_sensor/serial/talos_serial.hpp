#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l1_sensor/talos/talos_config.hpp"
#include "l1_sensor/talos/talos_reader.hpp"
#include "l5_control/serial_command.hpp"

#include <Eigen/Geometry>

#include <chrono>
#include <memory>
#include <optional>

namespace L1Sensor {

// 仿真器侧的"串口"替代：从 Talos 共享内存读云台状态，把控制命令写回。
class TalosSerial {
public:
  TalosSerial(
    std::shared_ptr<talos::TalosReader> reader,
    talos::TalosSimConfig config);

  [[nodiscard]] std::optional<RobotState> latestState() const;

  // 返回 R_world_barrel（OpenCV 相机系 → 世界系），供 L3 使用。
  [[nodiscard]] std::optional<Eigen::Quaterniond> gimbalPoseAt(
    std::chrono::steady_clock::time_point timestamp) const;

  // 写回仿真器云台命令；distance_m <= 0 时使用配置默认值。
  void updateCommand(
    const L5Control::SerialCommand& command, double distance_m);

  void stop() {}

private:
  std::shared_ptr<talos::TalosReader> reader_;
  talos::TalosSimConfig config_;
};

}  // namespace L1Sensor
