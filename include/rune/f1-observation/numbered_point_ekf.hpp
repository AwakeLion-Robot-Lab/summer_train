#pragma once

#include "rune/f6_telemetry/types.hpp"

namespace Rune {

class NumberedPointEkf {
public:
  void update(const Keypoints& blade, DetectedContours& output, cv::Size image_size,
              TimePoint now, double max_gap_s, double time_constant_s);

private:
  struct PointState {
    cv::Point2d position{};
    double variance{0};
    double innovation_variance{0};
    TimePoint measured_at{};
    unsigned observations{0};
  };

  static constexpr std::size_t gaps = 8, fan = 6;
  std::array<PointState, gaps+fan> points_{};
  TimePoint previous_time_{};
  TimePoint hub_measured_at_{};
  cv::Point2d previous_radial_{};
  double previous_scale_{0};
  double previous_diameter_{0};
};

}  // namespace Rune
