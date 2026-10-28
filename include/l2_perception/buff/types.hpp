#pragma once

#include <array>
#include <opencv2/core/types.hpp>
#include <vector>

namespace L2Perception::Buff {

// 网络四点沿周向排列，但不等同于传统 PnP 的十字臂端点。
struct ModelDetection {
  cv::Rect2f box;
  std::array<cv::Point2f, 4> keypoints;
  cv::Point2f center;
  float confidence{};
  int class_id{}; // 模型元数据：BR/BW/RR/RW。
  bool active{}; // BW/RW 已激活，BR/RR 待打。
};

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
  // 保留网络原始检测用于显示；几何精修失败的框仍可看见，但不送入跟踪。
  std::vector<ModelDetection> model_detections;
};

}  // namespace L2Perception::Buff
