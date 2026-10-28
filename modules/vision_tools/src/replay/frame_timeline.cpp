#include "vision_tools/replay/frame_timeline.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vision_tools {

TimingSource parseTimingSource(const std::string& text)
{
  if (text == "auto") return TimingSource::Auto;
  if (text == "video") return TimingSource::Video;
  if (text == "pose") return TimingSource::Pose;
  throw std::invalid_argument("timing-source must be auto, video, or pose");
}

double FrameTimeline::fps() const
{
  if (capture_seconds.size() < 2) return video_fps;
  return static_cast<double>(capture_seconds.size() - 1) /
         capture_seconds.back();
}

double FrameTimeline::secondsAt(int index) const
{
  if (index < 0) throw std::out_of_range("negative frame index");
  if (capture_seconds.empty()) return static_cast<double>(index) / video_fps;
  if (index < static_cast<int>(capture_seconds.size())) {
    return capture_seconds[static_cast<std::size_t>(index)];
  }
  return capture_seconds.back() +
         static_cast<double>(index - static_cast<int>(capture_seconds.size()) + 1) / fps();
}

int FrameTimeline::frameAt(double seconds) const
{
  if (!capture_seconds.empty()) {
    return std::max(0, static_cast<int>(std::upper_bound(
      capture_seconds.begin(), capture_seconds.end(), seconds) -
      capture_seconds.begin()) - 1);
  }
  const double index = std::clamp(
    std::floor(seconds * video_fps), 0.0,
    static_cast<double>(std::numeric_limits<int>::max()));
  return static_cast<int>(index);
}

FrameTimeline loadFrameTimeline(const std::filesystem::path& input,
                                double video_fps, int total_frames,
                                TimingSource source)
{
  if (!std::isfinite(video_fps) || video_fps <= 0.0) {
    throw std::invalid_argument("video FPS must be positive and finite");
  }
  FrameTimeline timeline{.video_fps = video_fps};
  if (source == TimingSource::Video) return timeline;

  auto pose_path = input;
  pose_path.replace_extension(".txt");
  if (!std::filesystem::exists(pose_path)) {
    if (source == TimingSource::Pose) {
      throw std::runtime_error("pose file not found: " + pose_path.string());
    }
    return timeline;
  }

  std::ifstream pose_file{pose_path};
  if (!pose_file) {
    throw std::runtime_error("cannot read pose file: " + pose_path.string());
  }
  std::string line;
  double first_seconds = 0.0;
  while (std::getline(pose_file, line)) {
    std::istringstream row{line};
    double seconds, w, x, y, z;
    std::string extra;
    if (!(row >> seconds >> w >> x >> y >> z) || (row >> extra) ||
        !std::isfinite(seconds) || !std::isfinite(w) || !std::isfinite(x) ||
        !std::isfinite(y) || !std::isfinite(z)) {
      throw std::runtime_error("invalid pose line " +
                               std::to_string(timeline.capture_seconds.size() + 1));
    }
    if (timeline.capture_seconds.empty()) first_seconds = seconds;
    const double relative = seconds - first_seconds;
    if (relative < 0.0 || (!timeline.capture_seconds.empty() &&
                           relative <= timeline.capture_seconds.back())) {
      throw std::runtime_error("pose times must increase strictly");
    }
    timeline.capture_seconds.push_back(relative);
  }
  if (timeline.capture_seconds.empty()) {
    throw std::runtime_error("pose file is empty: " + pose_path.string());
  }
  if (total_frames > 0 &&
      timeline.capture_seconds.size() != static_cast<std::size_t>(total_frames)) {
    throw std::runtime_error("video and pose frame counts differ: " + pose_path.string());
  }
  return timeline;
}

}  // namespace vision_tools
