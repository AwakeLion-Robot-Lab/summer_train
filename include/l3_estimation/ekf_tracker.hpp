#pragma once

#include "l3_estimation/types.hpp"

#include <vector>

namespace L3Estimation {

class EkfTracker {
public:
  explicit EkfTracker(int robot_id = -1);

  void predict(TimePoint timestamp);
  void update(const std::vector<ArmorObservation>& observations);
  void reset();

  [[nodiscard]] const TargetState& state() const noexcept;
  [[nodiscard]] bool expired(TimePoint now) const noexcept;

private:
  TargetState state_;
  TimePoint last_seen_{};
};

}  // namespace L3Estimation
