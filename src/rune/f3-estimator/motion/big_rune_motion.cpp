// Adapted from Alliance-Algorithm/rmcs_auto_aim_v2, RuneEnergyFitter (MIT).
// See ../ALLIANCE-LICENSE and ../README.md.
#include "rune/f3-estimator/estimator.hpp"
#include <Eigen/QR>
#include <cmath>

namespace Rune {

std::optional<PhaseFit> MotionFitter::fitSine() const {
  if (samples_.size()<8 || std::chrono::duration<double>(samples_.back().time-
      samples_.front().time).count()<config_.min_fit_s) return {};
  const TimePoint reference=samples_.back().time;
  const Eigen::Index n=samples_.size();
  Eigen::MatrixXd design(n,4);
  Eigen::VectorXd times(n),weights(n),y(n);
  for (Eigen::Index i=0;i<n;++i) {
    times[i]=std::chrono::duration<double>(samples_[i].time-reference).count();
    weights[i]=std::sqrt(std::exp2(times[i]/config_.weight_half_life_s));
    design(i,0)=weights[i]; design(i,1)=times[i]*weights[i];
    y[i]=samples_[i].phase*weights[i];
  }
  const double sum_weights=weights.squaredNorm();
  PhaseFit best;
  for (unsigned step=0;step<=config_.frequency_steps;++step) {
    const double omega=config_.frequency_min_rad_s+step*(config_.frequency_max_rad_s-
      config_.frequency_min_rad_s)/config_.frequency_steps;
    for (Eigen::Index i=0;i<n;++i) {
      design(i,2)=std::cos(omega*times[i])*weights[i];
      design(i,3)=std::sin(omega*times[i])*weights[i];
    }
    const auto qr=design.colPivHouseholderQr();
    if (qr.rank()!=4) continue;
    const Eigen::Vector4d coefficients=qr.solve(y);
    if (!coefficients.allFinite()) continue;
    const double cost=(y-design*coefficients).squaredNorm()/sum_weights;
    if (cost<best.cost) best={reference,coefficients[0],coefficients[1],
      coefficients[2],coefficients[3],omega,cost};
  }
  return std::isfinite(best.cost) ? std::optional<PhaseFit>(best) : std::nullopt;
}

Motion EstimatorStage::updateBigMotion(const Pose& pose) {
  const auto linear=fitter_.fitLinear(),sine=fitter_.fitSine();
  const double ratio=selected_model_==MotionModel::Sine ? 0.98 : config_.fit.sine_switch_cost_ratio;
  const bool sine_valid=sine && sine->amplitude()>=config_.fit.min_sine_amplitude_rad_s &&
    sine->amplitude()<=config_.fit.max_sine_amplitude_rad_s &&
    std::abs(sine->v)>=sine->amplitude() &&
    std::abs(sine->v)+sine->amplitude()<=config_.max_speed_rad_s &&
    (!linear || sine->cost<linear->cost*ratio);
  const auto fit=sine_valid ? sine : linear;
  Motion motion;
  if (fit) {
    selected_model_=sine_valid ? MotionModel::Sine : MotionModel::Linear;
    motion=fittedMotion(pose,*fit,selected_model_);
  } else motion.pose=pose;
  motion.linear_cost_rad2=linear ? linear->cost : std::numeric_limits<double>::infinity();
  motion.sine_cost_rad2=sine ? sine->cost : std::numeric_limits<double>::infinity();
  return motion;
}

} // namespace Rune
