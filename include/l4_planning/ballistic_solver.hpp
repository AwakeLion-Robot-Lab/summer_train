#pragma once

namespace L4Planning {

class BallisticSolver {
public:
  double solvePitch(double distance, double height, double bullet_speed) const;
};

}  // namespace L4Planning
