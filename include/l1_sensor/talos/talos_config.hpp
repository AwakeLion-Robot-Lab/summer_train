#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/robot_state.hpp"

#include <chrono>
#include <optional>
#include <string>

namespace L1Sensor::talos {

struct TalosConfig {
  std::string shm_dir{"/tmp"};
  EnemyColor enemy_color{EnemyColor::Red};
  WorkMode mode{WorkMode::AutoAim};
  double bullet_speed{25.0};
  bool shoot_enable{false};
  std::chrono::milliseconds timeout{50};
  double default_distance_m{1.0};
  double max_yaw_jump_deg{30.0};
  double max_pitch_jump_deg{45.0};
  std::optional<CameraCalibration> calibration;
};

TalosConfig loadTalosConfig(const std::string& path);

} // namespace L1Sensor::talos
