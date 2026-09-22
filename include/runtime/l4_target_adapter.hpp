#pragma once

#include "l3_estimation/armor/target_estimator.hpp"
#include "l3_estimation/target_state.hpp"

#include <optional>

namespace runtime {

// Copy the L3 filter into L4's snapshot so planning can predict on a copy.
[[nodiscard]] std::optional<L3Estimation::TargetState> toL4TargetState(
  const std::optional<L3Estimation::TrackedTarget>& target) noexcept;

}  // namespace runtime
