#pragma once

#include "l3_estimation/armor/target_estimator.hpp"
#include "l3_estimation/target_state.hpp"

#include <optional>

namespace runtime {

// Convert the target branch's mutable 13-state EKF object into the preserved
// L4 planner's immutable 11-state input snapshot.
[[nodiscard]] std::optional<L3Estimation::TargetState> toL4TargetState(
  const std::optional<L3Estimation::TrackedTarget>& target) noexcept;

}  // namespace runtime
