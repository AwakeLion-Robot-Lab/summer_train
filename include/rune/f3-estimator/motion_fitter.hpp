#pragma once
#include "rune/types.hpp"
#include <deque>
#include <limits>

namespace Rune {
struct MotionFitConfig {
  double window_s{6.0};
  double min_fit_s{1.5};
  double weight_half_life_s{3.0};
  unsigned max_samples{1200};
  double frequency_min_rad_s{1.80};
  double frequency_max_rad_s{2.20};
  unsigned frequency_steps{40};
  double min_sine_amplitude_rad_s{0.6};
  double max_sine_amplitude_rad_s{1.5};
  double max_fit_rms_rad{0.06};
  double sine_switch_cost_ratio{0.9};
};

// phase(t) = C + v*t + A*cos(omega*t) + B*sin(omega*t).
// t is relative to reference, the latest sample at fitting time.
struct PhaseFit {
  TimePoint reference{};
  double C{0},v{0},A{0},B{0},omega{0};
  double cost{std::numeric_limits<double>::infinity()};
  [[nodiscard]] double phase(double t) const;
  [[nodiscard]] double speed(double t) const;
  [[nodiscard]] double delta(double t,double horizon) const;
  [[nodiscard]] double amplitude() const;
};
class MotionFitter {
public:
  explicit MotionFitter(MotionFitConfig config = {});
  void reset() { samples_.clear(); }
  void push(TimePoint time,double phase);
  [[nodiscard]] std::optional<PhaseFit> fitLinear() const;
  [[nodiscard]] std::optional<PhaseFit> fitSine() const;
  [[nodiscard]] std::size_t size() const { return samples_.size(); }
private:
  struct Sample { TimePoint time; double phase; };
  MotionFitConfig config_;
  std::deque<Sample> samples_;
};
} // namespace Rune
