#pragma once

#include "rune/f6_telemetry/types.hpp"

#include <cstddef>

namespace Rune {

// 记录当前帧的轮廓检测结果。
struct DetectorDiagnostics {
  std::size_t contour_count{0}, target_candidates{0}, center_candidates{0};
  std::size_t arm_matches{0}, geometry_rejected{0};
  std::size_t scut_gap_points{0}, scut_fan_points{0}, contour_passes{0};
  std::size_t detections{0};
};

// 将原图坐标中的关键点填入 Frame::detections，供 Processor 使用。
class IKeypointDetector {
public:
  virtual ~IKeypointDetector() = default;
  [[nodiscard]] virtual std::vector<Keypoints> detect(const cv::Mat& bgr) = 0;
  [[nodiscard]] virtual std::vector<Keypoints> detect(const cv::Mat& bgr, Mode mode) {
    (void)mode;
    return detect(bgr);
  }
  [[nodiscard]] virtual std::optional<DetectorDiagnostics> lastDiagnostics() const {
    return std::nullopt;
  }
  [[nodiscard]] virtual std::vector<Keypoints> detect(const cv::Mat& bgr, Mode mode, Color color) {
    (void)color;
    return detect(bgr, mode);
  }
  [[nodiscard]] virtual std::vector<Keypoints> detect(const Frame& frame) {
    return detect(frame.bgr, frame.mode, frame.color);
  }
};

// 验证检测得到的全图关键点，并提取共享原图存储的只读候选区域。
class DetectionStage {
public:
  explicit DetectionStage(double roi_scale = 2.0) : roi_scale_(roi_scale) {}
  [[nodiscard]] std::optional<Region> run(const Frame& frame) const;
  // 包含作为辅助观测的已激活扇叶，候选区域保持源图像存储有效。
  [[nodiscard]] std::vector<Region> runAll(const Frame& frame) const;

private:
  double roi_scale_;
};

}  // namespace Rune
