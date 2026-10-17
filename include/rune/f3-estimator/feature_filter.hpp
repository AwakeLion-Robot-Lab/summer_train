#pragma once
#include "rune/types.hpp"

namespace Rune {
struct FeatureFilterConfig {
  double gate_squared{9.21}; // 2D chi-square, 99%
  double observation_variance_px2{4.0};
  double center_process_variance_m2_s{0.0025};
  double normal_process_variance_rad2_s{0.0025};
  double angular_acceleration_variance{4.0};
  double max_gap_s{0.25};
  double max_center_jump_m{0.3};
  double max_normal_jump_rad{0.45};
  double max_speed_rad_s{6.0};
};

class FeatureFilter {
public:
  explicit FeatureFilter(Geometry geometry, FeatureFilterConfig config = {});
  [[nodiscard]] std::optional<Pose> update(const Pose& seed,
      const std::vector<Observation>& observations, const Frame& frame);
  void reset();
private:
  Geometry geometry_;
  FeatureFilterConfig config_;
  std::optional<FilterState> state_;
  TimePoint previous_time_{};
  std::uint64_t world_frame_id_{0};
  int target_sector_{-1};
  Eigen::Vector3d up_{Eigen::Vector3d::UnitZ()};
};
} // namespace Rune
