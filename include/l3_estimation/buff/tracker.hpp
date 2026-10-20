#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/buff/types.hpp"
#include "l3_estimation/buff/target_estimator.hpp"

#include <Eigen/Geometry>
#include <memory>
#include <optional>
#include <span>

namespace L3Estimation::Buff {

class Tracker {
public:
  Tracker(const L1Sensor::CameraCalibration& calibration, Config config = {});

  void reset() noexcept;
  [[nodiscard]] std::optional<State> update(
    std::span<const L2Perception::Buff::Icon> icons,
    std::span<const L2Perception::Buff::Bullseye> bullseyes,
    const Eigen::Quaterniond& q_world_camera,
    const Eigen::Vector3d& t_world_camera,
    TimePoint timestamp);

  [[nodiscard]] const Addition& addition() const noexcept { return addition_; }
  [[nodiscard]] bool hasModel() const noexcept { return estimator_ != nullptr; }
  // 最近一次收到图像观测的时间，用于 Runtime 的 100 ms 旧目标保护。
  [[nodiscard]] TimePoint lastTimestamp() const noexcept { return stamp_; }

private:
  L1Sensor::CameraCalibration calibration_;
  Config config_;
  std::unique_ptr<TargetEstimator> estimator_;
  TimePoint stamp_{};
  TimePoint corrected_stamp_{};
  Addition addition_{};
};

}  // namespace L3Estimation::Buff
