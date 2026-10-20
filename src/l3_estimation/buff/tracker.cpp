#include "l3_estimation/buff/tracker.hpp"

#include <chrono>
#include <cmath>

namespace L3Estimation::Buff {

Tracker::Tracker(const L1Sensor::CameraCalibration& calibration, Config config)
: calibration_(calibration), config_(config)
{
}

void Tracker::reset() noexcept
{
  estimator_.reset();
  stamp_ = TimePoint{};
  corrected_stamp_ = TimePoint{};
  addition_ = {};
}

std::optional<State> Tracker::update(
  std::span<const L2Perception::Buff::Icon> icons,
  std::span<const L2Perception::Buff::Bullseye> bullseyes,
  const Eigen::Quaterniond& q_world_camera,
  const Eigen::Vector3d& t_world_camera,
  TimePoint timestamp)
{
  if (!estimator_) {
    if (icons.empty() || bullseyes.empty()) return std::nullopt;
    auto candidate = std::make_unique<TargetEstimator>(config_, calibration_);
    candidate->updateTransform(q_world_camera, t_world_camera);
    if (!candidate->init(icons, bullseyes, timestamp)) return std::nullopt;
    estimator_ = std::move(candidate);
    stamp_ = timestamp;
    corrected_stamp_ = timestamp;
    addition_ = estimator_->addition();
    return estimator_->state();
  }

  estimator_->updateTransform(q_world_camera, t_world_camera);

  // 源 RuneTracker 在没有观测时不推进滤波器，只让外层的 100 ms 旧帧门限
  // 决定是否还能使用上一帧结果。这样不会把缺帧误当作一帧真实测量。
  if (!icons.empty() || !bullseyes.empty()) {
    double dt = std::chrono::duration<double>(timestamp - stamp_).count();
    if (!std::isfinite(dt) || dt < 0.0) dt = 0.0;
    estimator_->predict(dt, timestamp);
    stamp_ = timestamp;
    if (estimator_->correct(icons, bullseyes)) corrected_stamp_ = timestamp;
  }

  addition_ = estimator_->addition();
  if (timestamp - corrected_stamp_ > std::chrono::duration_cast<TimePoint::duration>(
        std::chrono::duration<double>{config_.timeout_seconds}) ||
      estimator_->diverged()) {
    reset();
    return std::nullopt;
  }
  return estimator_->state();
}

}  // namespace L3Estimation::Buff
