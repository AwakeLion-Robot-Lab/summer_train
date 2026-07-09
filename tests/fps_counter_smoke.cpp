#include "l6_telemetry/fps_counter.hpp"

#include <chrono>
#include <cmath>
#include <iostream>

int main()
{
  using Clock = L6Telemetry::FpsCounter::Clock;
  using namespace std::chrono_literals;

  L6Telemetry::FpsCounter counter(1s);
  const auto start = Clock::time_point{};
  counter.reset(start);

  for (int i = 1; i <= 150; ++i) {
    counter.update(start + std::chrono::milliseconds(i * 10));
  }

  if (std::abs(counter.fps() - 100.0) > 0.001) {
    std::cerr << "FPS mismatch: " << counter.fps() << '\n';
    return 1;
  }

  if (counter.frames() != 150) {
    std::cerr << "Frame count mismatch: " << counter.frames() << '\n';
    return 2;
  }

  counter.reset(start);
  if (counter.fps() != 0.0 || counter.frames() != 0) {
    std::cerr << "Counter reset failed\n";
    return 3;
  }

  std::cout << "FpsCounter smoke test passed\n";
  return 0;
}
