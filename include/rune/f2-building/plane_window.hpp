#pragma once
#include "rune/types.hpp"
#include <deque>

namespace Rune {
// Accumulates only in a caller-provided fixed world frame. No hidden assumption
// that an IMU orientation also provides vehicle translation.
class PlaneWindow {
public:
  void reset() { samples_.clear(); }
  void update(Pose& pose, const Frame& frame, const Geometry& geometry);
private:
  struct Sample {
    TimePoint time;
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Matrix3d basis;
    std::uint64_t frame_id;
  };
  std::deque<Sample> samples_;
};
} // namespace Rune
