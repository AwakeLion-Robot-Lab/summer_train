#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/buff/types.hpp"
#include "l1_sensor/serial/robot_state.hpp"

#include <opencv2/core/mat.hpp>

namespace L2Perception::Buff {

struct DetectorConfig {
  double min_distance{1.7};
  double max_distance{5.0};
  double max_perspective_deg{60.0};
  double active_threshold{0.2};
  double match_threshold{0.50};
};

class Detector {
public:
  Detector() = default;
  explicit Detector(DetectorConfig config) : config_(config) {}

  void setCalibration(const L1Sensor::CameraCalibration& calibration);
  void setEnemyColor(L1Sensor::EnemyColor color) noexcept { color_ = color; }

  [[nodiscard]] Detections detect(const cv::Mat& image) const;

  const DetectorConfig& config() const noexcept { return config_; }

private:
  DetectorConfig config_{};
  L1Sensor::CameraCalibration calibration_{};
  L1Sensor::EnemyColor color_{L1Sensor::EnemyColor::Unknown};
  bool calibration_ready_{false};
};

}  // namespace L2Perception::Buff
