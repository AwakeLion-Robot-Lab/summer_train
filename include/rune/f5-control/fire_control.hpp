#pragma once

#include "rune/f4-planning/planning.hpp"
#include "sensor/serial/serial_command.hpp"

namespace Rune {

struct FireControlConfig {
  bool shoot_enable{false};
  double max_yaw_error_rad{0.01};
  double max_pitch_error_rad{0.01};
  double aim_settle_s{0.03};
  double idle_duration_s{0.8};
  double shoot_duration_s{0.12};
  double min_shot_interval_s{0.3};
  double max_plan_age_s{0.15};
  double command_jump_threshold_rad{0.15};
  // Optional legacy single-shot latch. Timed mode can retry a missed blade.
  bool require_rearm{false};
};

class ControlStage {
public:
  explicit ControlStage(FireControlConfig config = {});
  [[nodiscard]] Command update(const AimPlan& plan, const ControlInput& control);
  // Interrupt timing and settling; retain shot cooldown and manual latch.
  void suspend() noexcept;
  void reset() noexcept;
  void rearm() noexcept;
  [[nodiscard]] std::optional<L1Sensor::SerialCommand> safeHold() const;
  [[nodiscard]] static std::optional<L1Sensor::SerialCommand> makeSerialCommand(
      const Command& command);

private:
  FireControlConfig config_;
  std::optional<Mode> mode_;
  std::uint64_t epoch_{0};
  int sector_{-1};
  TimePoint cycle_started_{}, aligned_since_{}, last_shot_{}, previous_command_{};
  std::int64_t fired_cycle_{-1};
  bool latched_{false};
  std::optional<L1Sensor::SerialCommand> last_command_;
};

} // namespace Rune
