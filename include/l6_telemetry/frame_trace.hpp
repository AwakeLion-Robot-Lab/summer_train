#pragma once

#include <chrono>

namespace L6Telemetry {

struct FrameTrace {
  int frame_id = 0;
  std::chrono::steady_clock::time_point capture_time{};
};

}  // namespace L6Telemetry
