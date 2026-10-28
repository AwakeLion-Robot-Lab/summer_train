#include "vision_tools/replay/frame_timeline.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

int main()
{
  const auto path = std::filesystem::temp_directory_path() /
    "vision_tools_timeline_smoke.avi";
  auto pose = path;
  pose.replace_extension(".txt");
  {
    std::ofstream out(pose);
    out << "0.000 1 0 0 0\n0.020 1 0 0 0\n0.050 1 0 0 0\n";
  }
  const auto timeline = vision_tools::loadFrameTimeline(
    path, 90.0, 3, vision_tools::TimingSource::Auto);
  const bool okay = timeline.hasCaptureTimes() &&
    std::abs(timeline.secondsAt(2) - 0.05) < 1e-9 &&
    timeline.frameAt(0.03) == 1 &&
    std::abs(timeline.fps() - 40.0) < 1e-9;
  std::filesystem::remove(pose);
  const auto fallback = vision_tools::loadFrameTimeline(
    path, 90.0, 3, vision_tools::TimingSource::Auto);
  if (!okay || fallback.hasCaptureTimes() || fallback.frameAt(0.03) != 2) {
    std::cerr << "frame timeline smoke failed\n";
    return 1;
  }
  std::cout << "Frame timeline smoke test passed\n";
  return 0;
}
