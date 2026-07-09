#pragma once

#include "l2_perception/detector.hpp"

namespace L2Perception {

class LightbarDetector : public Detector {
public:
  std::vector<Detection> detect(const cv::Mat& image) override;
};

}  // namespace L2Perception
