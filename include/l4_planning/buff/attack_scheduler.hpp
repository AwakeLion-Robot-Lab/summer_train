#pragma once

#include <chrono>

namespace L4Planning::Buff {

class AttackScheduler {
public:
  AttackScheduler(double idle_seconds, double shoot_seconds) noexcept;

  void reset() noexcept;
  [[nodiscard]] bool started() const noexcept { return started_; }
  [[nodiscard]] bool tick(std::chrono::steady_clock::time_point now);
  void start(std::chrono::steady_clock::time_point now) noexcept;

private:
  double idle_seconds_;
  double shoot_seconds_;
  bool started_{false};
  int phase_{0};
  std::chrono::steady_clock::time_point phase_start_{};
};

}  // namespace L4Planning::Buff
