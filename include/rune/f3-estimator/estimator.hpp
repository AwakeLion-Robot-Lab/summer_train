#pragma once

#include "rune/types.hpp"

#include <memory>
#include <utility>

namespace Rune {

struct EstimatorConfig {
  double max_gap_s{0.25};
  double max_speed_rad_s{6.0};
  unsigned min_samples{3};
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
                          std::unique_ptr<IBigMotionModel> big_model = {})
    : config_(config), big_model_(std::move(big_model)) {}
  [[nodiscard]] Motion update(const Pose& pose, Mode mode);
  void reset();

private:
  EstimatorConfig config_;
  std::optional<Pose> previous_;
  double filtered_speed_{0.0};
  unsigned samples_{0};
  std::optional<Mode> mode_;
  std::unique_ptr<IBigMotionModel> big_model_;
};

}  // namespace Rune
