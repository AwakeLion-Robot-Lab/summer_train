#include "l4_planning/buff/attack_scheduler.hpp"

#include <chrono>

namespace L4Planning::Buff {

AttackScheduler::AttackScheduler(double idle_seconds, double shoot_seconds) noexcept
: idle_seconds_(idle_seconds), shoot_seconds_(shoot_seconds)
{
}

void AttackScheduler::reset() noexcept
{
  started_ = false;
  phase_ = 0;
  phase_start_ = {};
}

void AttackScheduler::start(std::chrono::steady_clock::time_point now) noexcept
{
  if (started_) return;
  started_ = true;
  phase_ = 0;
  phase_start_ = now;
}

bool AttackScheduler::tick(std::chrono::steady_clock::time_point now)
{
  if (!started_) return false;
  const double elapsed = std::chrono::duration<double>(now - phase_start_).count();
  const double duration = phase_ == 0 ? idle_seconds_ : shoot_seconds_;
  if (elapsed >= duration) {
    phase_ = phase_ == 0 ? 1 : 0;
    phase_start_ = now;
  }
  return phase_ == 1;
}

}  // namespace L4Planning::Buff
