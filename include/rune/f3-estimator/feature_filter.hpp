#pragma once
#include "rune/f6_telemetry/types.hpp"

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
  int big_sector_hold_ms{150};
  int big_sector_confirmation_ms{75};
};

class FeatureFilter {
public:
  struct TargetPrediction {
    Eigen::Vector3d observed_direction;
    Eigen::Vector3d predicted_direction;
  };
  struct GimbalPrediction {
    Eigen::Vector3d direction;
    int sector;
  };
  struct TargetPriorProjection {
    int sector{0};
    cv::Matx33d homography_target_to_image{cv::Matx33d::eye()};
    cv::RotatedRect projected_ellipse;
    cv::Point2f center_pixel{};
    cv::Point2f outward_direction{};
    cv::Point2f hub_pixel{};
    std::array<cv::Point2f, 8> gap_pixels{};
    std::array<double, 8> gap_search_sigma_px{};
  };

  explicit FeatureFilter(Geometry geometry, FeatureFilterConfig config = {}, bool follow_only = false);
  [[nodiscard]] std::optional<Pose> update(const Pose& seed,
      const std::vector<Observation>& observations, const Frame& frame);
  void updateReobservedPhase(const Frame& frame);
  bool updateReobservedTargetGaps(
      const Frame& frame, int sector,
      const std::array<std::optional<cv::Point2f>, 8>& gap_points,
      const std::array<float, 8>& gap_variances);
  void noteObservationGap();
  [[nodiscard]] std::vector<TargetPriorProjection> predictTargetPriors(const Frame& frame) const;
  [[nodiscard]] int targetSector() const noexcept { return target_sector_; }
  [[nodiscard]] std::optional<TargetPrediction> predictTarget(
      const Frame& frame, TimePoint at, int sector) const;
  [[nodiscard]] std::optional<GimbalPrediction> predictGimbal(
      const Frame& frame, TimePoint at, bool center_only) const;
  [[nodiscard]] bool hasHistory() const noexcept { return state_.has_value(); }
  void reset();
private:
  Geometry geometry_;
  FeatureFilterConfig config_;
  bool follow_only_;
  std::optional<FilterState> state_;
  TimePoint previous_time_{};
  std::uint64_t world_frame_id_{0};
  int target_sector_{-1};
  int pending_sector_{-1};
  TimePoint pending_since_{};
  TimePoint target_seen_at_{};
  Eigen::Vector3d up_{Eigen::Vector3d::UnitZ()};
  std::optional<Eigen::Quaterniond> previous_world_orientation_;
  Eigen::Vector3d previous_world_origin_{Eigen::Vector3d::Zero()};
};
} // namespace Rune
