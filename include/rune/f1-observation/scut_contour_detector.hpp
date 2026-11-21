#pragma once

#include "rune/f1-observation/detection.hpp"

namespace YAML { class Node; }

namespace Rune {

struct ScutContourConfig {
  int red_threshold{40}, blue_threshold{15};
  int fallback_threshold_offset{10}; // Retry a stricter mask if the first has no match; 0 disables.
  int close_kernel{5}; // Odd diameter in [1, 19]; 1 disables LED reconnection.
  double min_target_area{60}, max_target_area{100000};
  double min_center_area{20}, max_center_area{2500};
  double min_center_rect_fill{0.70};
  double min_center_target_area_ratio{0.03};
  double max_target_axis_ratio{3.0};
  double min_ellipse_fill{0.8};
  double max_ellipse_residual{0.10};
  double radius_ratio{0.15 / 0.70}; // Target radius / R-to-target distance.
  double radius_ratio_tolerance{0.25};
  double max_arm_offset_ratio{0.24};
  double min_arm_color_ratio{0.25}; // (enemy-other)/(enemy+other); reject neutral reflections.
  int max_features{128};
  [[nodiscard]] static ScutContourConfig fromYaml(
      const YAML::Node& node, const YAML::Node& geometry);
};

// SCUT-style color / contour-tree / gap matching, adapted to the local five
// anchors. No network, inferred R, hidden-window state or actuator dependency.
class ScutContourDetector final : public IKeypointDetector {
public:
  explicit ScutContourDetector(ScutContourConfig config = {});
  [[nodiscard]] std::vector<Keypoints> detect(const cv::Mat& bgr) override;
  [[nodiscard]] std::vector<Keypoints> detect(const cv::Mat& bgr, Mode mode) override;
  [[nodiscard]] std::vector<Keypoints> detect(const cv::Mat& bgr, Mode mode, Color color) override;
  [[nodiscard]] std::optional<DetectorDiagnostics> lastDiagnostics() const override {
    return diagnostics_;
  }
private:
  [[nodiscard]] std::vector<Keypoints> detectAtThreshold(
      const cv::Mat& bgr, Mode mode, Color color, int threshold);
  ScutContourConfig config_;
  std::optional<DetectorDiagnostics> diagnostics_;
};

} // namespace Rune
