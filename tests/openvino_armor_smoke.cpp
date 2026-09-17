#include "l2_perception/armor/armor_detector.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

cv::Point toPixel(const cv::Point2f& point)
{
  return {static_cast<int>(std::lround(point.x)), static_cast<int>(std::lround(point.y))};
}

void drawDetections(
  cv::Mat& image,
  const std::vector<L2Perception::Armor>& detections,
  double display_fps,
  double l2_fps)
{
  cv::putText(image, cv::format("display %.1f FPS | L2 %.1f FPS", display_fps, l2_fps),
              {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2, cv::LINE_AA);

  for (const auto& detection : detections) {
    const cv::Scalar color = detection.color == L2Perception::ArmorColor::Red
                               ? cv::Scalar{0, 0, 255}
                               : detection.color == L2Perception::ArmorColor::Blue
                                   ? cv::Scalar{255, 0, 0}
                                   : cv::Scalar{0, 255, 255};
    const char* color_name = detection.color == L2Perception::ArmorColor::Red
                               ? "red"
                               : detection.color == L2Perception::ArmorColor::Blue ? "blue" : "unknown";

    // 画网络（或精修后）的四角点，坐标已从 letterbox 还原到原图。
    for (std::size_t index = 0; index < detection.corners.size(); ++index) {
      const auto& start = detection.corners[index];
      const auto& end = detection.corners[(index + 1) % detection.corners.size()];
      cv::line(image, toPixel(start), toPixel(end), color, 2, cv::LINE_AA);
    }

    const std::string label = color_name + std::string{" class="}
                              + std::to_string(detection.class_id) + " conf="
                              + std::to_string(detection.confidence).substr(0, 4);
    cv::putText(image, label, toPixel(detection.corners.front()) + cv::Point{0, -6},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    // 不给模型参数时跑 auto_aim.yaml 里真正配置的那一个；给了参数则只换整板
    // 模型，layout 按模型输出名认，精修参数仍取 YAML。
    const auto runtime_config = runtime::loadConfig("config/auto_aim.yaml");
    const std::filesystem::path model_path = argc < 2
      ? runtime_config.model_path
      : std::filesystem::path{argv[1]};
    // 第三个可选参数用于同口径比较 CPU/GPU；不传时保持稳定的 CPU 默认值。
    const std::string device = argc >= 4 ? std::string{argv[3]} : std::string{"CPU"};
    const bool show_window = argc >= 5 && std::string_view{argv[4]} == "--show";

    cv::Mat image;
    cv::VideoCapture video;
    bool video_mode = false;
    if (argc >= 3) {
      image = cv::imread(argv[2], cv::IMREAD_COLOR);
      if (image.empty()) {
        // imread() 读不了时再按视频打开；这样同一个参数位置可直接传 jpg/png/mp4。
        video.open(argv[2]);
        require(video.isOpened(), "failed to open input image or video: " + std::string{argv[2]});
        require(video.read(image) && !image.empty(), "input video contains no readable frame");
        video_mode = true;
      }
    } else {
      // 没有测试图片时仍使用真实模型推理；黑图只作为相机帧输入，不伪造模型输出。
      image = cv::Mat(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
    }

    // 与实机同一个工厂组装：整板模型 + 角点精修，量到的就是实机 L2 的整帧耗时。
    // detect() 不给 light_roi，侧边灯条那一路不参与计时。
    runtime::AutoAimConfig detector_config = runtime_config;
    detector_config.inference.model_path = model_path;
    detector_config.inference.device = device;
    const L2Perception::ArmorDetector armor_detector =
      runtime::makeDetector(detector_config, argc >= 2);
    require(armor_detector.ready(), "ArmorDetector did not accept the loaded backend");
    const auto detect_frame = [&](const cv::Mat& frame) {
      return armor_detector.detect(frame);
    };

    constexpr int warmup_frames = 5;

    for (int frame = 0; frame < warmup_frames; ++frame) {
      (void)detect_frame(image);
    }

    std::size_t processed_frames = 0;
    std::size_t frames_with_detections = 0;
    std::size_t total_detections = 0;
    std::size_t max_detections_per_frame = 0;
    std::array<std::size_t, 3> color_counts{};  // Red、Blue、Unknown。
    std::array<std::size_t, 9> class_counts{};  // G、1、2、3、4、5、O、Bs、Bb。
    double total_l2_ms = 0.0;
    bool display_open = show_window;
    auto previous_display_time = std::chrono::steady_clock::now();
    double display_fps = 0.0;
    if (show_window) {
      cv::namedWindow("armor inference", cv::WINDOW_NORMAL);
    }

    const auto measure_frame = [&](const cv::Mat& frame) {
      // 统计“相邻两次显示开始”的间隔，包含读视频、推理、绘制和窗口刷新。
      const auto display_begin = std::chrono::steady_clock::now();
      const double display_interval_ms =
        std::chrono::duration<double, std::milli>(display_begin - previous_display_time).count();
      if (display_interval_ms > 0.0) {
        const double instantaneous_fps = 1000.0 / display_interval_ms;
        // 简单低通滤波，避免每帧文字因系统调度产生剧烈跳动。
        display_fps = display_fps == 0.0 ? instantaneous_fps
                                          : display_fps * 0.9 + instantaneous_fps * 0.1;
      }
      previous_display_time = display_begin;

      const auto begin = std::chrono::steady_clock::now();
      const auto detections = detect_frame(frame);
      const auto end = std::chrono::steady_clock::now();
      const double l2_ms = std::chrono::duration<double, std::milli>(end - begin).count();

      total_l2_ms += l2_ms;
      ++processed_frames;
      total_detections += detections.size();
      if (!detections.empty()) {
        ++frames_with_detections;
      }
      max_detections_per_frame = std::max(max_detections_per_frame, detections.size());

      for (const auto& detection : detections) {
        switch (detection.color) {
        case L2Perception::ArmorColor::Red:
          ++color_counts[0];
          break;
        case L2Perception::ArmorColor::Blue:
          ++color_counts[1];
          break;
        case L2Perception::ArmorColor::Unknown:
          ++color_counts[2];
          break;
        }
        if (detection.class_id >= 0
            && detection.class_id < static_cast<int>(class_counts.size())) {
          ++class_counts[static_cast<std::size_t>(detection.class_id)];
        }
      }

      if (display_open) {
        // 不修改推理输入帧；可视化使用独立副本，避免影响下一帧或统计结果。
        cv::Mat visualization = frame.clone();
        drawDetections(visualization, detections, display_fps, 1000.0 / l2_ms);
        cv::imshow("armor inference", visualization);
        // pollKey() 不主动等待下一帧，窗口按“推理完成即刷新”的速度播放。
        const int key = cv::pollKey();
        display_open = key != 27 && key != 'q' && key != 'Q';
      }

      return display_open || !show_window;
    };

    if (video_mode) {
      // 第一帧已被读取用于预热；先正式统计它，再顺序处理视频剩余全部帧。
      bool keep_running = measure_frame(image);
      while (keep_running && video.read(image)) {
        if (!image.empty()) {
          keep_running = measure_frame(image);
        }
      }
    } else {
      constexpr int measured_frames = 20;
      for (int frame = 0; frame < measured_frames && measure_frame(image); ++frame) {
      }
    }

    require(processed_frames != 0, "no frame was processed");
    const double average_ms = total_l2_ms / static_cast<double>(processed_frames);

    std::cout << "OpenVINO armor smoke passed\nmodel " << model_path.string() << "\ndevice "
              << device << "\naverage full L2 time: " << average_ms
              << " ms\nprocessed frames: " << processed_frames
              << "\nframes with detections: " << frames_with_detections
              << "\ntotal detections: " << total_detections
              << "\nmax detections per frame: " << max_detections_per_frame

              << "\ncolors: red=" << color_counts[0] << ", blue=" << color_counts[1]
              << ", unknown=" << color_counts[2]
              << "\nclasses [G,1,2,3,4,5,O,Bs,Bb]: ";
    for (std::size_t index = 0; index < class_counts.size(); ++index) {
      if (index != 0) {
        std::cout << ',';
      }
      std::cout << class_counts[index];
    }
    std::cout << '\n';

    if (video_mode) {
      const double fps = video.get(cv::CAP_PROP_FPS);
      const double declared_frames = video.get(cv::CAP_PROP_FRAME_COUNT);
      std::cout << "video metadata: fps=" << fps << ", declared_frames=" << declared_frames << '\n';
    }
    if (show_window) {
      cv::destroyWindow("armor inference");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "OpenVINO armor smoke failed: " << error.what() << '\n';
    return 1;
  }
}
