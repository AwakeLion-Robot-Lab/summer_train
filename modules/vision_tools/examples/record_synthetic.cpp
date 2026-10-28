#include "vision_tools/recording/recorder.hpp"

#include <chrono>
#include <iostream>
#include <thread>

#include <opencv2/core.hpp>

int main(int argc, char* argv[])
{
  vision_tools::RecorderConfig config;
  config.enabled = true;
  config.fps = 30.0;
  config.output_dir = argc > 1 ? argv[1] : "records";
  vision_tools::Recorder recorder(config);
  const auto started = std::chrono::steady_clock::now();
  int accepted = 0;
  for (int index = 0; index < 60; ++index) {
    cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(index * 4, 40, 160));
    const auto captured = std::chrono::steady_clock::now();
    if (recorder.record(frame, Eigen::Quaterniond::Identity(), captured)) {
      ++accepted;
    }
    std::this_thread::sleep_until(started + std::chrono::milliseconds(34 * (index + 1)));
  }
  recorder.stop();
  std::cout << "Recorded " << accepted << " synthetic frames\n";
  return accepted > 0 ? 0 : 1;
}
