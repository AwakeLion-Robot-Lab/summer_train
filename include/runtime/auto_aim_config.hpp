#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l3_estimation/ekf_tracker.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"

namespace runtime {

struct AutoAimConfig {
  L3Estimation::ArmorDimensions armor;
  L3Estimation::EkfTrackerConfig tracker;
  L4Planning::PlanConfig plan;
  L5Control::FireConfig fire;

  [[nodiscard]] bool fireReady(
    const L1Sensor::CameraCalibration& calibration) const noexcept
  {
    return fire.shoot_enable && calibration.barrelExtrinsicsReady() &&
           plan.fireDelayReady() && fire.parametersReady();
  }
};

}  // namespace runtime
