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
};

// Transfers the previous measurement with verified local image motion, then
// corrects residual localization noise with the current NN observation. Does
// not synthesize detections, change validity or extrapolate unobserved points.
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
  TimePoint previous_time_{};
  Mode previous_mode_{Mode::Small};
  Color previous_color_{Color::Blue};
  ImageRefinerDiagnostics diagnostics_;
};
} // namespace Rune
