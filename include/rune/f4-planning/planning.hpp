#pragma once

#include "rune/types.hpp"
#include <array>

namespace Rune {

struct PlanningConfig {
  // Compatibility settings used by Processor's default f5 configuration.
  bool shoot_enable{false};
  double gravity_m_s2{9.80665};
  double max_observation_age_s{0.15};
  double max_yaw_error_rad{0.01};
  double max_pitch_error_rad{0.01};
  bool enable_air_resistance{false};
  double linear_drag_coefficient{0.0};
  double small_warmup_s{3.0};
  double big_warmup_s{6.0};
  double inactive_timeout_s{0.1};
  double max_prediction_s{2.0};
  int max_iterations{12};
  double flight_time_tolerance_s{0.0002};
  double yaw_offset_rad{0.0};
  double pitch_offset_rad{0.0};
  bool center_hold{true};
  // Reuse the existing TinyMPC tuning. Ballistics above apply to every sample.
  L4Planning::PlannerConfig mpc{};
};

enum class MpcStatus { Disabled, Applied, InvalidReference, NotConverged };
struct MpcDiagnostics {
  MpcStatus status{MpcStatus::Disabled};
  int yaw_iterations{0}, pitch_iterations{0};
  double yaw_primal_residual{0}, pitch_primal_residual{0};
  double yaw_dual_residual{0}, pitch_dual_residual{0};
};

// Shared trajectory/sample representation can be passed to existing consumers.
struct AimPlan : L4Planning::AimPlan {
  Mode mode{Mode::Small};
  Status status{Status::MotionNotReady};
  TimePoint captured_at{};
  std::uint64_t tracking_epoch{0};
  bool center_hold{false};
  Eigen::Vector3d center_aim{Eigen::Vector3d::Zero()};
  double ballistic_yaw_rad{0.0}, ballistic_pitch_rad{0.0};
  double prediction_horizon_s{0.0};
  bool used_air_resistance{false};
  MpcDiagnostics mpc_diagnostics;
};

class PlanningStage {
public:
  explicit PlanningStage(PlanningConfig config = {});
  [[nodiscard]] AimPlan plan(const Motion& motion, Mode mode,
                              const ControlInput& control);
  void reset() noexcept;

private:
  struct BladeState { bool inactive{false}; TimePoint seen_at{}; };
  PlanningConfig config_;
  std::array<BladeState,5> blades_{};
  std::optional<Mode> mode_;
  TimePoint started_at_{}, previous_capture_{};
  std::uint64_t epoch_{0};
  int selected_sector_{-1};
};

} // namespace Rune
