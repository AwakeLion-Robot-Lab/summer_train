#pragma once

#include <array>
#include <opencv2/core/types.hpp>
#include <vector>

namespace L2Perception::Buff {

// 源 RuneDetector 的二维输出。靶心 corners 是四个“十字臂端点”，不是外接框角点。
struct Icon {
  cv::Point2f center{};
  double score{0.0};
};

struct Bullseye {
  cv::Point2f center{};
  std::array<cv::Point2f, 4> corners{};
  // 保持源工程语义：RuneModel 用 !active 判断“未激活”。
  bool active{false};
  double score{0.0};
};

struct Detections {
  std::vector<Icon> icons;
  std::vector<Bullseye> bullseyes;
};

}  // namespace L2Perception::Buff
