#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>

namespace L1Sensor {

struct FramePacket {
  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  int frame_id = 0;
};

}  // namespace L1Sensor
