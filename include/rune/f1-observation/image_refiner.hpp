#pragma once

#include "rune/f6_telemetry/types.hpp"
#include "rune/f1-observation/numbered_point_ekf.hpp"
#include <opencv2/core/ocl.hpp>
#include <filesystem>
#include <string>

namespace YAML { class Node; }

namespace Rune {
struct ImageRefinerConfig {
  bool enabled{false};
  double max_gap_s{0.2};
  double measurement_time_constant_s{0.08};
  double min_correlation{0.98};
  bool use_opencl{false};
  std::string opencl_device;
  bool track_numbered_points{false};
  bool reobserve_target_rings{false};
  bool refine_inactive_arm_edges{false};
  [[nodiscard]] static ImageRefinerConfig load(const std::filesystem::path& yaml);
  [[nodiscard]] static ImageRefinerConfig fromYaml(const YAML::Node& node);
};

struct ImageRefinerDiagnostics {
  std::size_t refined_r{0}, refined_targets{0};
  std::size_t hub_registrations{0};
  std::size_t rejected_target_rings{0};
  std::size_t refined_feature_targets{0}, refined_fan_points{0}, rejected_feature_points{0};
  std::size_t gpu_ecc_batches{0}, gpu_ecc_accepted{0};
  double ring_reobservation_ms{0};
  double inactive_arm_ms{0};
};

// Transfers the previous measurement with verified local image motion, then
// corrects residual localization noise in detected anchors and SCUT features.
// Numbered SCUT points can additionally carry display-only predictions.
// Image-verified rings that change size without corresponding image motion lose target validity.
class ImageRefiner {
public:
  explicit ImageRefiner(ImageRefinerConfig config = {});
  void update(Frame& frame);
  void reset();
  [[nodiscard]] ImageRefinerDiagnostics diagnostics() const { return diagnostics_; }
  [[nodiscard]] std::string registrationDevice() const;
private:
  struct OutlineTrack {
    DisplayBladeOutline shape;
    std::optional<cv::Point2f> hub;
    TimePoint observed_at;
    bool history_only{false};
  };
  void updateDisplayOutlines(Frame& frame, const cv::Mat& gray);

  ImageRefinerConfig config_;
  cv::ocl::OpenCLExecutionContext opencl_;
  cv::Mat previous_gray_;
  std::vector<cv::Mat> previous_flow_pyramid_;
  int previous_flow_levels_{-1};
  cv::UMat previous_gpu_gray_;
  std::vector<Keypoints> previous_detections_;
  // Association and registration windows must not depend on filtered output.
  std::vector<Keypoints> previous_raw_detections_;
  std::vector<NumberedPointEkf> numbered_point_tracks_;
  std::vector<OutlineTrack> outline_tracks_;
  cv::Mat outline_gray_;
  TimePoint outline_time_{};
  TimePoint previous_time_{};
  Mode previous_mode_{Mode::Small};
  Color previous_color_{Color::Blue};
  ImageRefinerDiagnostics diagnostics_;
};
} // namespace Rune
