#pragma once

#include "rune/f6_telemetry/types.hpp"

namespace Rune {

struct RuneCenterFeature {
  cv::Point2f position;
  // 轮廓为空时，位置由同一机关上可见的叶片推算。
  std::vector<cv::Point> contour;
};

struct RuneTargetFeature {
  cv::RotatedRect ellipse;
  std::vector<cv::Point> contour;
  // 顺序为上、左、右、下。
  std::array<cv::Point2f, 4> anchors;
  std::array<std::optional<cv::Point2f>, 8> gap_points;
  std::array<float, 8> gap_point_variances{};
};

struct RuneFanCorner {
  int id;
  cv::Point2f pixel;
};

struct RuneFanFeature {
  bool activated;
  // 同一扇叶上可见灯条的外接范围。
  cv::Rect bounds;
  std::vector<cv::Point> contour;
  // 已激活扇叶完整时为 0～5，部分可见时为三个外端 0～2。
  std::vector<RuneFanCorner> corners;
  std::optional<InactiveArrowAxis> inactive_arrow_axis{};
};

struct RuneBladeFeature {
  std::optional<RuneTargetFeature> target;
  RuneFanFeature fan;
  float confidence;
};

struct RuneGroup {
  Color color;
  RuneCenterFeature center;
  std::vector<RuneBladeFeature> blades;
};

} // namespace Rune
