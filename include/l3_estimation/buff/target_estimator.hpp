#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/buff/types.hpp"
#include "l3_estimation/buff/types.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <memory>
#include <optional>
#include <span>

namespace L3Estimation::Buff {

class TargetEstimator {
public:
  TargetEstimator(Config config, const L1Sensor::CameraCalibration& calibration);
  ~TargetEstimator();

  TargetEstimator(const TargetEstimator&) = delete;
  TargetEstimator& operator=(const TargetEstimator&) = delete;
  TargetEstimator(TargetEstimator&&) noexcept = default;
  TargetEstimator& operator=(TargetEstimator&&) noexcept = default;

  void updateTransform(const Eigen::Quaterniond& q_world_camera,
                       const Eigen::Vector3d& t_world_camera);

  [[nodiscard]] bool init(
    std::span<const L2Perception::Buff::Icon> icons,
    std::span<const L2Perception::Buff::Bullseye> bullseyes,
    TimePoint timestamp);
  void predict(double dt, TimePoint timestamp) noexcept;
  [[nodiscard]] bool correct(
    std::span<const L2Perception::Buff::Icon> icons,
    std::span<const L2Perception::Buff::Bullseye> bullseyes) noexcept;

  [[nodiscard]] bool converge() const noexcept { return true; }
  [[nodiscard]] bool diverged() const noexcept;
  [[nodiscard]] State state() const noexcept;
  [[nodiscard]] const Addition& addition() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace L3Estimation::Buff
