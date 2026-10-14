#pragma once

#include "rune/f1-observation/detection.hpp"
#include "rune/f1-observation/reobservation.hpp"
#include "rune/f2-building/builder.hpp"
#include "rune/f3-estimator/estimator.hpp"
#include "rune/f4-planning/planning.hpp"

namespace Rune {

class Processor {
public:
  explicit Processor(Geometry geometry, EstimatorConfig estimator = {},
                     PlanningConfig planning = {},
                     std::unique_ptr<IBigMotionModel> big_model = {});
  [[nodiscard]] Command process(const Frame& frame,
                                 const ControlInput& control);
  // Call after external confirmation that the previous blade was activated
  // and a different target is selected; clears the one-shot latch.
  void rearm() noexcept { planning_.resetFireLatch(); }
  void reset();

private:
  DetectionStage detection_;
  ObservationStage observation_;
  BuildingStage building_;
  EstimatorStage estimator_;
  PlanningStage planning_;
  std::optional<Mode> previous_mode_;
};

}  // namespace Rune
