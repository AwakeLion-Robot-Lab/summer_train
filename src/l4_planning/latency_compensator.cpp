#include "l4_planning/latency_compensator.hpp"

namespace L4Planning {

LatencyCompensator::LatencyCompensator(double latency_seconds)
  : latency_seconds_(latency_seconds)
{
}

double LatencyCompensator::latency() const
{
  return latency_seconds_;
}

}  // namespace L4Planning
