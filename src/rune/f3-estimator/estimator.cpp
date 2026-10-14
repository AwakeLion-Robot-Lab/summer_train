#include "rune/f3-estimator/estimator.hpp"

#include <cmath>
#include <numbers>

namespace Rune {

void EstimatorStage::reset() {
  if (big_model_) big_model_->reset();
  previous_.reset();
  mode_.reset();
  filtered_speed_ = 0;
  samples_ = 0;
}

Motion EstimatorStage::update(const Pose& pose, Mode mode) {
  Motion motion;
  motion.pose = pose;
  if (mode_ && *mode_ != mode) reset();
  mode_ = mode;

  // Big rune uses a separately fitted phase model supplied by the caller.
  if (mode == Mode::Big) {
    if (!big_model_) return motion;
    motion = big_model_->update(pose);
    motion.pose = pose;
    motion.ready &= static_cast<bool>(motion.phase_delta_rad);
    return motion;
  }

  if (!previous_) {
    previous_ = pose;
    samples_ = 1;
    return motion;
  }
  const double dt = std::chrono::duration<double>(
    pose.captured_at - previous_->captured_at).count();
  if (!std::isfinite(dt) || dt <= 0 || dt > config_.max_gap_s) {
    reset();
    mode_ = mode;
    previous_ = pose;
    samples_ = 1;
    return motion;
  }
  const double delta = std::remainder(
    pose.phase_rad - previous_->phase_rad, 2.0 * std::numbers::pi);
  const double speed = delta / dt;
  if (!std::isfinite(speed) || std::abs(speed) > config_.max_speed_rad_s ||
      (pose.center_barrel - previous_->center_barrel).norm() > 0.3 ||
      pose.normal_barrel.dot(previous_->normal_barrel) < 0.9) {
    reset();
    mode_ = mode;
    previous_ = pose;
    samples_ = 1;
    return motion;
  }
  filtered_speed_ = samples_ < 2 ? speed : 0.65 * filtered_speed_ + 0.35 * speed;
  ++samples_;
  previous_ = pose;
  motion.angular_speed_rad_s = filtered_speed_;
  motion.ready = samples_ >= config_.min_samples &&
                 std::abs(filtered_speed_) > 0.05;
  return motion;
}

}  // namespace Rune
