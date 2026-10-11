#include "l6_telemetry/fps_counter.hpp"

namespace L6Telemetry {

FpsCounter::FpsCounter(std::chrono::milliseconds sample_window)
  : sample_window_(sample_window > std::chrono::milliseconds::zero() ? sample_window
                                                                     : std::chrono::seconds(1)),
    last_sample_time_(Clock::now())
{
}

double FpsCounter::update()
{
  return update(Clock::now());
}

double FpsCounter::update(Clock::time_point now)
{
  ++frames_in_window_;
  ++total_frames_;

  const auto elapsed = now - last_sample_time_;
  if (elapsed >= sample_window_) {
    const auto seconds = std::chrono::duration<double>(elapsed).count();
    fps_ = static_cast<double>(frames_in_window_) / seconds;
    frames_in_window_ = 0;
    last_sample_time_ = now;
  }

  return fps_;
}

double FpsCounter::fps() const
{
  return fps_;
}

uint64_t FpsCounter::frames() const
{
  return total_frames_;
}

void FpsCounter::reset()
{
  reset(Clock::now());
}

void FpsCounter::reset(Clock::time_point now)
{
  last_sample_time_ = now;
  frames_in_window_ = 0;
  total_frames_ = 0;
  fps_ = 0.0;
}

}  // namespace L6Telemetry
