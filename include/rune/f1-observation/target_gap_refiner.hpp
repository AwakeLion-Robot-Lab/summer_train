#pragma once

#include "rune/f6_telemetry/types.hpp"

#include <array>
#include <optional>

namespace Rune {

struct TargetGapRefinementResult {
  cv::RotatedRect refined_ellipse;
  bool ellipse_refined{false};
  std::array<std::optional<cv::Point2f>, 8> gap_points{};
  std::array<float, 8> gap_point_variances{};
  double phase_offset_rad{0.0};
  int measured_spokes{0};
  int measured_gaps{0};
};

// 在椭圆归一化极坐标或靶面单应性坐标系下，直接对原图 BGR 色差通道采样并做亚像素精修。
// 仅当具备当帧图像边缘和对比度依据时产出关键点与逐点像素方差。
[[nodiscard]] TargetGapRefinementResult refineTargetGapsFromImage(
    const cv::Mat& bgr, Color color, const cv::RotatedRect& initial_ellipse,
    cv::Point2f outward_direction,
    const std::optional<cv::Matx33d>& homography_target_to_image = std::nullopt,
    const std::array<double, 8>* prior_sigma_px = nullptr,
    bool refine_outer_ellipse = true);

// 在规范化靶面平面内，对已配对的前后帧 8 点观测执行 Huber 加权二维刚体修正，
// 剔除单点离群值并输出经刚体约束精修后的关键点与方差。
void refineGapPointsRigidHuber(
    std::array<std::optional<cv::Point2f>, 8>& current_points,
    std::array<float, 8>& current_variances,
    const std::array<std::optional<cv::Point2f>, 8>& predicted_points,
    const std::array<float, 8>& predicted_variances,
    const cv::RotatedRect& target_ellipse, cv::Point2f outward_direction,
    cv::Size image_size, float alpha, std::size_t& rejected_count);

}  // namespace Rune
