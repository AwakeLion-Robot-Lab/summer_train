#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace L2Perception {

struct Detection {
  std::vector<cv::Point2f> corners;
  int class_id = -1;
  float confidence = 0.0F;
};

class Detector {
public:
  virtual ~Detector() = default;
  virtual std::vector<Detection> detect(const cv::Mat& image) = 0;
};

}  // namespace L2Perception
