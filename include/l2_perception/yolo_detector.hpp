#pragma once

#include "l2_perception/detector.hpp"

#include <string>

namespace L2Perception {

class YoloDetector : public Detector {
public:
  explicit YoloDetector(std::string model_path);
  std::vector<Detection> detect(const cv::Mat& image) override;

private:
  std::string model_path_;
};

}  // namespace L2Perception
