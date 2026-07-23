#include "l4_planning/latency_compensator.hpp"

#include "l4_planning/types.hpp"

namespace L4Planning {

LatencyCompensator::LatencyCompensator(LatencyConfig config)
{
  (void)config;
}

LatencyResult LatencyCompensator::calculate(const Delay& delay) const noexcept
{
  LatencyResult result;
  if (delay.camera_timestamp == TimePoint{} ||
      delay.command_timestamp < delay.camera_timestamp ||
      delay.fire_delay < 0.0) {
    return result;
  }

  result.delay = delay;
  result.confidence = 1.0;
  result.valid = true;
  return result;
}

}  // namespace L4Planning
