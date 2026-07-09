#include "l2_perception/yolo_detector.hpp"

#include <utility>

namespace L2Perception {

YoloDetector::YoloDetector(std::string model_path)
  : model_path_(std::move(model_path))
{
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& image)
{
  (void)image;
  return {};
}

}  // namespace L2Perception
