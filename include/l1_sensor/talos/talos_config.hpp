#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/robot_state.hpp"

#include <chrono>
#include <optional>
#include <string>

namespace L1Sensor::talos {

// Talos 仿真器接入配置，字段缺失时全部保留默认值。
struct TalosSimConfig {
  std::string shm_dir{"/tmp"};
  EnemyColor enemy_color{EnemyColor::Red};
  WorkMode mode{WorkMode::AutoAim};
  double bullet_speed{25.0};
  bool shoot_enable{false};
  std::chrono::milliseconds timeout{50};
  double default_distance_m{1.0};
  double gt_match_gate_m{0.6};
  double report_interval_s{1.0};

  // 可选：完整相机标定（内参 + 相机→枪管外参）。
  // 缺省时 TalosCamera 从共享内存 camera_info 构造内参，外参取单位阵。
  std::optional<CameraCalibration> calibration;
};

// 从相机配置 YAML 读取：calibration 节点 + talos 节点。
// YAML 缺失/解析错误时抛出异常。
TalosSimConfig loadTalosSimConfig(const std::string& config_path);

}  // namespace L1Sensor::talos
