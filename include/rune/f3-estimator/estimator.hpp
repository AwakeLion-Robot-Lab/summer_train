#pragma once

#include "rune/types.hpp"
#include "rune/f3-estimator/feature_filter.hpp"
#include "rune/f3-estimator/motion_fitter.hpp"

#include <memory>
#include <utility>

namespace Rune {

struct EstimatorConfig {
  double max_gap_s{0.25};
  double max_speed_rad_s{6.0};
  unsigned min_samples{3};
  MotionFitConfig fit{};
  FeatureFilterConfig filter{};
};

class IBigMotionModel {
public:
  virtual ~IBigMotionModel() = default;
  // Return a ready Motion only after sufficient fit quality. Its phase_delta_rad
  // must own a snapshot of fitted coefficients; do not capture mutable state.
  [[nodiscard]] virtual Motion update(const Pose& pose) = 0;
  virtual void reset() = 0;
};

class EstimatorStage {
public:
  explicit EstimatorStage(EstimatorConfig config = {},
                          std::unique_ptr<IBigMotionModel> big_model = {});
  [[nodiscard]] Motion update(const Pose& pose, Mode mode);
  void reset();

private:
  [[nodiscard]] Motion updateSmallMotion(const Pose& pose);
  [[nodiscard]] Motion updateBigMotion(const Pose& pose);
  [[nodiscard]] Motion fittedMotion(const Pose& pose, const PhaseFit& fit,
                                    MotionModel model) const;

  EstimatorConfig config_;
  std::optional<Pose> previous_;
  double continuous_phase_{0.0};
  unsigned samples_{0};
  std::optional<Mode> mode_;
  std::unique_ptr<IBigMotionModel> big_model_;
  MotionFitter fitter_;
  MotionModel selected_model_{MotionModel::Unready};
};

}  // namespace Rune
