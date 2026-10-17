#include "rune/f3-estimator/estimator.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace Rune {
EstimatorStage::EstimatorStage(EstimatorConfig config,std::unique_ptr<IBigMotionModel> big_model)
  : config_(config),big_model_(std::move(big_model)),fitter_(config.fit) {
  if (!std::isfinite(config.max_gap_s) || config.max_gap_s<=0 ||
      !std::isfinite(config.max_speed_rad_s) || config.max_speed_rad_s<=0 || config.min_samples<2)
    throw std::invalid_argument("Invalid rune estimator configuration");
}
void EstimatorStage::reset() {
  if (big_model_) big_model_->reset();
  previous_.reset(); mode_.reset(); fitter_.reset();
  continuous_phase_=0; samples_=0; selected_model_=MotionModel::Unready;
}
Motion EstimatorStage::update(const Pose& pose,Mode mode) {
  Motion motion; motion.pose=pose;
  if ((mode_ && *mode_!=mode) || pose.history_reset) reset();
  mode_=mode;
  // An injected model overrides only the big-rune path.
  if (mode==Mode::Big && big_model_) {
    motion=big_model_->update(pose); motion.pose=pose;
    motion.ready &= static_cast<bool>(motion.phase_delta_rad);
    motion.model=motion.ready ? MotionModel::External : MotionModel::Unready;
    return motion;
  }
  const auto start=[&] {
    previous_=pose; samples_=1;
    continuous_phase_=pose.filter_state ? pose.filter_state->x[5] : pose.phase_rad;
    fitter_.push(pose.captured_at,continuous_phase_);
  };
  if (!std::isfinite(pose.phase_rad) || !pose.center_barrel.allFinite() ||
      !pose.normal_barrel.allFinite() ||
      (pose.filter_state && !pose.filter_state->x.allFinite())) { reset(); return motion; }
  if (!previous_) { start(); return motion; }
  const double dt=std::chrono::duration<double>(pose.captured_at-previous_->captured_at).count();
  if (!std::isfinite(dt) || dt<=0 || dt>config_.max_gap_s ||
      bool(pose.filter_state)!=bool(previous_->filter_state)) {
    reset(); mode_=mode; start(); return motion;
  }
  double delta;
  if (pose.filter_state) delta=pose.filter_state->x[5]-continuous_phase_;
  else {
    delta=std::remainder(pose.phase_rad-previous_->phase_rad,2*std::numbers::pi);
    if (pose.target_switched) delta=std::remainder(delta,2*std::numbers::pi/5);
  }
  // The feature filter gates pose changes in its tracking frame. Direct Pose
  // callers retain the old geometric discontinuity checks.
  if (!std::isfinite(delta) || std::abs(delta/dt)>config_.max_speed_rad_s ||
      (!pose.filter_state && ((pose.center_barrel-previous_->center_barrel).norm()>0.3 ||
       pose.normal_barrel.dot(previous_->normal_barrel)<0.9))) {
    reset(); mode_=mode; start(); return motion;
  }
  continuous_phase_+=delta; previous_=pose; ++samples_;
  fitter_.push(pose.captured_at,continuous_phase_);
  return mode==Mode::Big ? updateBigMotion(pose) : updateSmallMotion(pose);
}

Motion EstimatorStage::fittedMotion(const Pose& pose,const PhaseFit& fit,
                                   MotionModel model) const {
  Motion motion; motion.pose=pose;
  motion.model=model; motion.fit_cost_rad2=fit.cost;
  motion.sine_frequency_rad_s=model==MotionModel::Sine ? fit.omega : 0;
  motion.angular_speed_rad_s=fit.speed(0);
  motion.ready=samples_>=config_.min_samples &&
    fit.cost<=config_.fit.max_fit_rms_rad*config_.fit.max_fit_rms_rad &&
    std::abs(motion.angular_speed_rad_s)>0.05 &&
    std::abs(motion.angular_speed_rad_s)<=config_.max_speed_rad_s;
  // The intercept cancels: prediction starts at the current filtered pose.
  motion.phase_delta_rad=[snapshot=fit](double horizon) -> std::optional<double> {
    if (!std::isfinite(horizon) || horizon<0) return {};
    const double delta=snapshot.delta(0,horizon);
    return std::isfinite(delta) ? std::optional<double>(delta) : std::nullopt;
  };
  motion.phase_delta_signed_rad=[snapshot=fit](double horizon) -> std::optional<double> {
    if (!std::isfinite(horizon)) return {};
    const double delta=snapshot.delta(0,horizon);
    return std::isfinite(delta) ? std::optional<double>(delta) : std::nullopt;
  };
  return motion;
}
} // namespace Rune
