#pragma once

#include "rune/types.hpp"
#include <filesystem>

namespace YAML { class Node; }

namespace Rune {
struct ImageRefinerConfig {
  bool enabled{false};
  double max_gap_s{0.2};
  double measurement_time_constant_s{0.08};
  double min_correlation{0.98};
  [[nodiscard]] static ImageRefinerConfig load(const std::filesystem::path& yaml);
  [[nodiscard]] static ImageRefinerConfig fromYaml(const YAML::Node& node);
};

struct ImageRefinerDiagnostics {
  std::size_t refined_r{0}, refined_targets{0};
  std::size_t rejected_target_rings{0};
  std::size_t refined_feature_targets{0}, refined_fan_points{0}, rejected_feature_points{0};
};

// Transfers the previous measurement with verified local image motion, then
// corrects residual localization noise in network anchors and SCUT features. Does
// not synthesize detections or extrapolate unobserved points. Image-verified
// rings that change size without corresponding image motion lose target validity.
class ImageRefiner {
public:
  explicit ImageRefiner(ImageRefinerConfig config = {});
  void update(Frame& frame);
  void reset();
  [[nodiscard]] ImageRefinerDiagnostics diagnostics() const { return diagnostics_; }
private:
  ImageRefinerConfig config_;
  cv::Mat previous_gray_;
  std::vector<Keypoints> previous_detections_;
  // Association and registration windows must not depend on filtered output.
  std::vector<Keypoints> previous_raw_detections_;
  std::vector<std::array<cv::Point2f,4>> previous_coarse_filtered_;
  std::vector<cv::Point2f> previous_ring_center_corrections_;
  std::vector<TimePoint> previous_ring_precision_times_;
  TimePoint previous_time_{};
  Mode previous_mode_{Mode::Small};
  Color previous_color_{Color::Blue};
  ImageRefinerDiagnostics diagnostics_;
};
} // namespace Rune
