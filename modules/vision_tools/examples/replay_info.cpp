#include "vision_tools/replay/frame_timeline.hpp"

#include <iostream>
#include <opencv2/videoio.hpp>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "usage: replay_info <video.avi>\n";
    return 2;
  }
  cv::VideoCapture video(argv[1]);
  if (!video.isOpened()) {
    std::cerr << "cannot open video\n";
    return 1;
  }
  const auto timeline = vision_tools::loadFrameTimeline(
    argv[1], video.get(cv::CAP_PROP_FPS),
    static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT)));
  std::cout << "frames=" << video.get(cv::CAP_PROP_FRAME_COUNT)
            << " effective_fps=" << timeline.fps()
            << " timing=" << (timeline.hasCaptureTimes() ? "capture" : "video")
            << '\n';
  return 0;
}
