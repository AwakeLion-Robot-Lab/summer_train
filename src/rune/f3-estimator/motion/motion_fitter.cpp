// Adapted from Alliance-Algorithm/rmcs_auto_aim_v2, RuneEnergyFitter (MIT).
// See ../ALLIANCE-LICENSE and ../README.md. Time is recentered for numerical stability.
#include "rune/f3-estimator/motion_fitter.hpp"
#include <cmath>
#include <stdexcept>

namespace Rune {
double PhaseFit::phase(double t) const {
  return C+v*t+A*std::cos(omega*t)+B*std::sin(omega*t);
}
double PhaseFit::speed(double t) const {
  return v-omega*A*std::sin(omega*t)+omega*B*std::cos(omega*t);
}
double PhaseFit::delta(double t,double horizon) const {
  return v*horizon+A*(std::cos(omega*(t+horizon))-std::cos(omega*t))+
    B*(std::sin(omega*(t+horizon))-std::sin(omega*t));
}
double PhaseFit::amplitude() const { return omega*std::hypot(A,B); }
MotionFitter::MotionFitter(MotionFitConfig config) : config_(config) {
  for (double v : {config.window_s,config.min_fit_s,config.weight_half_life_s,
      config.frequency_min_rad_s,config.frequency_max_rad_s,
      config.max_sine_amplitude_rad_s,config.max_fit_rms_rad,config.sine_switch_cost_ratio})
    if (!std::isfinite(v) || v<=0) throw std::invalid_argument("Invalid rune fit configuration");
  if (config.min_fit_s>config.window_s || config.max_samples<8 || config.max_samples>100000 ||
      config.frequency_steps<1 || config.frequency_steps>1000 ||
      config.frequency_min_rad_s>=config.frequency_max_rad_s ||
      !std::isfinite(config.min_sine_amplitude_rad_s) || config.min_sine_amplitude_rad_s<0 ||
      config.min_sine_amplitude_rad_s>config.max_sine_amplitude_rad_s ||
      config.sine_switch_cost_ratio>1)
    throw std::invalid_argument("Inconsistent rune fit configuration");
}
void MotionFitter::push(TimePoint time,double phase) {
  if (!std::isfinite(phase)) return;
  if (!samples_.empty() && time<=samples_.back().time) return;
  samples_.push_back({time,phase});
  while (samples_.size()>config_.max_samples ||
      std::chrono::duration<double>(time-samples_.front().time).count()>config_.window_s)
    samples_.pop_front();
}
std::optional<PhaseFit> MotionFitter::fitLinear() const {
  if (samples_.size()<2 || std::chrono::duration<double>(samples_.back().time-
      samples_.front().time).count()<config_.min_fit_s) return {};
  PhaseFit fit; fit.reference=samples_.back().time;
  double sw=0,st=0,sy=0,stt=0,sty=0;
  for (const auto& s : samples_) {
    const double t=std::chrono::duration<double>(s.time-fit.reference).count();
    const double w=std::exp2(t/config_.weight_half_life_s);
    sw+=w; st+=w*t; sy+=w*s.phase; stt+=w*t*t; sty+=w*t*s.phase;
  }
  const double denominator=sw*stt-st*st;
  if (denominator<1e-12) return {};
  fit.v=(sw*sty-st*sy)/denominator; fit.C=(sy-fit.v*st)/sw;
  fit.cost=0;
  for (const auto& s : samples_) {
    const double t=std::chrono::duration<double>(s.time-fit.reference).count();
    const double error=s.phase-fit.phase(t);
    fit.cost+=std::exp2(t/config_.weight_half_life_s)*error*error/sw;
  }
  return std::isfinite(fit.cost) ? std::optional<PhaseFit>(fit) : std::nullopt;
}
} // namespace Rune
