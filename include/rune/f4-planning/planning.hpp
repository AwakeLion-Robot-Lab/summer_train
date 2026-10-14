#pragma once

#include "rune/types.hpp"

namespace Rune {

struct PlanningConfig {
  bool shoot_enable{false};
  double gravity_m_s2{9.80665};
  double max_observation_age_s{0.15};
  double max_yaw_error_rad{0.01};
  double max_pitch_error_rad{0.01};
};

class PlanningStage {
public:
  explicit PlanningStage(PlanningConfig config = {}) : config_(config) {}
  [[nodiscard]] Command plan(const Motion& motion, Mode mode,
                             const ControlInput& control);
  void resetFireLatch() noexcept { fired_ = false; }

private:
  PlanningConfig config_;
  bool fired_{false};
};

}  // namespace Rune
