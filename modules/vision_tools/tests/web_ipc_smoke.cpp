#include "vision_tools/web/web_cmd.hpp"
#include "vision_tools/web/web_sink.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include <opencv2/core.hpp>

int main()
{
  const auto directory = std::filesystem::temp_directory_path() /
    ("vision_tools_web_" + std::to_string(::getpid()));
  std::filesystem::create_directories(directory);
  const std::string prefix = (directory / "ipc").string();
  const std::string recording = (directory / "record").string();
  {
    L6Telemetry::WebSink sink(8, recording, 4, prefix);
    sink.writeFrame(cv::Mat(32, 32, CV_8UC3, cv::Scalar(0, 255, 0)), 17, 1.25);
    sink.writeSeries({{"t", 1.25}, {"aim", {{"yaw", 2.0}}}});
    sink.flush();
    if (!std::filesystem::exists(prefix + "_frame") ||
        !std::filesystem::exists(prefix + "_data.json") ||
        !std::filesystem::exists(recording + "/frames/frame_000000.jpg")) {
      std::cerr << "Web sink output missing\n";
      return 1;
    }
    std::ifstream index(recording + "/frames/index.jsonl");
    nlohmann::json row;
    index >> row;
    if (row["source_frame"] != 17 || row["t"] != 1.25) {
      std::cerr << "Web frame timeline invalid\n";
      return 2;
    }
  }
  {
    std::ofstream command(prefix + "_cmd.json");
    command << R"({"debug":{"overlay":true}})";
  }
  L6Telemetry::WebCmd cmd(prefix + "_cmd.json");
  if (!cmd.poll() || !cmd.has("debug.overlay")) {
    std::cerr << "Web command input invalid\n";
    return 3;
  }
  std::filesystem::remove_all(directory);
  return 0;
}
