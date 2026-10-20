#pragma once

#include "l3_estimation/buff/types.hpp"
#include "l4_planning/buff/types.hpp"
#include "l4_planning/buff/attack_scheduler.hpp"
#include "l4_planning/buff/trajectory_solver.hpp"

#include <optional>

namespace L4Planning::Buff {

class Planner {
public:
  explicit Planner(Config config = {});

  [[nodiscard]] Plan plan(
    const L3Estimation::Buff::State& state,
    TimePoint plan_time,
    BulletSpeed bullet_speed,
    bool to_now = true);

  void reset() noexcept;

private:
  Config config_;
  TrajectorySolver trajectory_;
  AttackScheduler scheduler_;
};

}  // namespace L4Planning::Buff
