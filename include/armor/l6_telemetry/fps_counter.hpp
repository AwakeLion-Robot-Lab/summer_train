#pragma once

#include <chrono>
#include <cstdint>

namespace L6Telemetry {

class FpsCounter {
public:
  using Clock = std::chrono::steady_clock;

  explicit FpsCounter(std::chrono::milliseconds sample_window = std::chrono::seconds(1));

  double update();
  double update(Clock::time_point now);
  double fps() const;
  uint64_t frames() const;
  void reset();
  void reset(Clock::time_point now);

private:
  std::chrono::milliseconds sample_window_;
  Clock::time_point last_sample_time_;
  uint64_t frames_in_window_ = 0;
  uint64_t total_frames_ = 0;
  double fps_ = 0.0;
};

}  // namespace L6Telemetry
