#pragma once

#include <Eigen/Core>
#include <optional>

namespace L4Planning::Buff {

struct TrajectoryOutput {
  double fly_time{0.0};
  double yaw{0.0};
  double pitch{0.0};
};

class TrajectorySolver {
public:
  [[nodiscard]] std::optional<TrajectoryOutput> solve(
    const Eigen::Vector3d& point, double bullet_speed) const;
};

}  // namespace L4Planning::Buff
