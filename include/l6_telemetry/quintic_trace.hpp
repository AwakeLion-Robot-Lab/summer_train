#pragma once

#include "l4_planning/armor/planner.hpp"

#include <nlohmann/json.hpp>

namespace L6Telemetry {

// 测试入口使用的数值遥测。NaN 表示本帧无定义，UDP 序列化为 null，
// CSV 写空单元格；不能将丢目标或切板处无定义的导数伪装成零。
class QuinticTrace {
public:
  QuinticTrace(L4Planning::BlendLimits limits, double yaw_speed, double pitch_speed);
  nlohmann::json update(
    L4Planning::TimePoint now, const L4Planning::Plan& plan,
    const L4Planning::PlannerDiagnostics& diagnostics,
    double measured_yaw, double measured_pitch);
  void reset();

private:
  struct Difference {
    bool has_position{false};
    bool has_velocity{false};
    double position{0.0};
    double velocity{0.0};
    double dt{0.0};
    L4Planning::AxisState update(double angle, double step);
  };
  L4Planning::BlendLimits limits_;
  double yaw_speed_;
  double pitch_speed_;
  bool started_{false};
  bool previous_valid_{false};
  L4Planning::TimePoint origin_{};
  L4Planning::TimePoint previous_time_{};
  L4Planning::TrajectorySampler previous_reference_;
  L4Planning::TrajectorySampler committed_after_;
  L4Planning::BlendSolution previous_segment_;
  L4Planning::TimePoint previous_segment_start_{};
  Difference raw_yaw_, raw_pitch_, planned_yaw_, planned_pitch_;
  double observed_seconds_{0.0};
  double blend_seconds_{0.0};
  std::uint64_t commits_{0}, late_commits_{0}, infeasible_commits_{0}, resets_{0};
};

}  // namespace L6Telemetry
