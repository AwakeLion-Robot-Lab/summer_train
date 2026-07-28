#pragma once

#include <Eigen/Core>

namespace L4Planning {

struct BallisticRequest {
  Eigen::Vector3d target_position_barrel{Eigen::Vector3d::Zero()};
  double bullet_speed{0.0};
  double gravity{9.80665};
};

struct BallisticSolution {
  double pitch{0.0};
  double yaw{0.0};
  double fly_time{0.0};
  bool valid{false};
};

class BallisticSolver {
public:
  double solvePitch(double distance, double height, double bullet_speed) const;
  [[nodiscard]] BallisticSolution solve(const BallisticRequest& request) const;
};

}  // namespace L4Planning
