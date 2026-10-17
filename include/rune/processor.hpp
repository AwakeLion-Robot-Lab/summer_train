#pragma once

#include "rune/f1-observation/detection.hpp"
#include "rune/f1-observation/reobservation.hpp"
#include "rune/f2-building/builder.hpp"
#include "rune/f3-estimator/estimator.hpp"
#include "rune/f4-planning/planning.hpp"
#include "rune/f5-control/fire_control.hpp"

namespace Rune {

class Processor {
public:
  explicit Processor(Geometry geometry, EstimatorConfig estimator = {},
                     PlanningConfig planning = {},
                     std::unique_ptr<IBigMotionModel> big_model = {},
                     std::optional<FireControlConfig> fire = {});
  [[nodiscard]] Command process(const Frame& frame,
                                 const ControlInput& control);
  [[nodiscard]] const std::optional<Motion>& lastMotion() const noexcept {
    return last_motion_;
  }
  [[nodiscard]] const std::optional<AimPlan>& lastPlan() const noexcept { return last_plan_; }
  [[nodiscard]] std::optional<L1Sensor::SerialCommand> safeHold() const {
    return control_.safeHold();
  }
  // Needed only when FireControlConfig::require_rearm is enabled.
  void rearm() noexcept { control_.rearm(); }
  void reset();

private:
  DetectionStage detection_;
  ObservationStage observation_;
  BuildingStage building_;
  FeatureFilter filter_;
  EstimatorStage estimator_;
  PlanningStage planning_;
  ControlStage control_;
  std::optional<Mode> previous_mode_;
  std::optional<Motion> last_motion_;
  std::optional<AimPlan> last_plan_;
};

}  // namespace Rune
