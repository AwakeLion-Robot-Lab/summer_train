#pragma once

#include "l4_planning/types.hpp"

namespace L4Planning {

struct LatencyConfig {
  [[nodiscard]] constexpr bool ready() const noexcept { return true; }
};

struct LatencyResult {
  Delay delay;
  bool valid{false};
};

class LatencyCompensator {
public:
  explicit LatencyCompensator(LatencyConfig config = {});

  [[nodiscard]] LatencyResult calculate(const Delay& delay) const noexcept;

private:
  LatencyConfig config_;
};

}  // namespace L4Planning
