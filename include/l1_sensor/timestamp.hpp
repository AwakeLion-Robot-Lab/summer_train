#pragma once

#include <chrono>
#include <cstdint>

namespace L1Sensor {

using SteadyTimePoint = std::chrono::steady_clock::time_point;

inline int64_t toMilliseconds(SteadyTimePoint timestamp)
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count();
}

}  // namespace L1Sensor
