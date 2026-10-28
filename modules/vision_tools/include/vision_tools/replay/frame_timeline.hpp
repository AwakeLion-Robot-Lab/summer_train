#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vision_tools {

enum class TimingSource { Auto, Video, Pose };

TimingSource parseTimingSource(const std::string& text);

// The legacy recorder writes a constant AVI FPS and exact per-frame capture
// times to an adjacent TXT file. Playback must use the latter when available.
struct FrameTimeline {
  double video_fps{30.0};
  std::vector<double> capture_seconds;

  [[nodiscard]] double fps() const;
  [[nodiscard]] double secondsAt(int index) const;
  [[nodiscard]] int frameAt(double seconds) const;
  [[nodiscard]] bool hasCaptureTimes() const noexcept
  {
    return !capture_seconds.empty();
  }
};

FrameTimeline loadFrameTimeline(const std::filesystem::path& input,
                                double video_fps, int total_frames,
                                TimingSource source = TimingSource::Auto);

}  // namespace vision_tools
