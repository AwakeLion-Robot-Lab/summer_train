#pragma once

#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"

namespace runtime {

struct AutoAimConfig {
  L3Estimation::ArmorConfig armor;
  L3Estimation::TrackerConfig tracker;
  L4Planning::PlanConfig plan;
  L5Control::FireConfig fire;

  [[nodiscard]] bool fireReady(
    const L3Estimation::AimCalibration& calibration) const noexcept
  {
    return fire.shoot_enable && calibration.fireReady() && plan.fireDelayReady() &&
           fire.parametersReady();
  }
};

}  // namespace runtime
