#include "rune/f1-observation/keypoint_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    std::filesystem::path config{"config/rune_detector.yaml"};
    std::string input, output;
    Rune::Mode mode = Rune::Mode::Small;
    int frames = 1;
    for (int arg = 1; arg < argc; ++arg) {
      const std::string flag = argv[arg];
      if (flag == "--help") {
        std::cout << "rune_detect --input IMAGE_OR_VIDEO [--mode small|big] "
                     "[--config YAML] [--frames N] [--output IMAGE.png]\n";
        return 0;
      }
      if (arg + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      const std::string value = argv[++arg];
      if (flag == "--input") input = value;
      else if (flag == "--output") output = value;
      else if (flag == "--config") config = value;
      else if (flag == "--frames") frames = std::stoi(value);
      else if (flag == "--mode") {
        if (value == "small") mode = Rune::Mode::Small;
        else if (value == "big") mode = Rune::Mode::Big;
        else throw std::invalid_argument("--mode must be small or big");
      } else throw std::invalid_argument("unknown argument: " + flag);
    }
    if (input.empty() || frames < 1)
      throw std::invalid_argument("specify --input IMAGE_OR_VIDEO and a positive --frames count");
    Rune::OpenVinoKeypointDetector detector(Rune::KeypointDetectorConfig::load(config));
    cv::Mat image = cv::imread(input);
    const bool still = !image.empty();
    cv::VideoCapture video;
    if (!still && !video.open(input)) throw std::runtime_error("cannot open input: " + input);
    cv::Mat annotated;
    int processed = 0, positive_frames = 0;
    std::size_t total_detections = 0;
    double elapsed_ms = 0;
    while (processed < frames) {
      if (!still && !video.read(image)) break;
      const auto start = std::chrono::steady_clock::now();
      const auto detections = detector.detect(image, mode);
      elapsed_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      ++processed;
      positive_frames += !detections.empty();
      total_detections += detections.size();
      std::cout << "frame=" << processed << " detections=" << detections.size() << '\n';
      if (!output.empty()) annotated = image.clone();
      for (const auto& detection : detections) {
        const std::array<cv::Point2f, 5> points{detection.top, detection.left,
          detection.right, detection.bottom, detection.center_r};
        std::cout << "  class=" << detection.class_id << " confidence=" << detection.confidence;
        for (const auto& point : points) std::cout << " (" << point.x << ',' << point.y << ')';
        std::cout << '\n';
        if (!output.empty()) {
          const auto color = detection.class_id == 0 ? cv::Scalar(0,255,0) : cv::Scalar(0,165,255);
          for (std::size_t p = 0; p < points.size(); ++p) {
            cv::circle(annotated, points[p], p == 4 ? 5 : 3, color, cv::FILLED);
            cv::putText(annotated, std::to_string(p), points[p] + cv::Point2f(5,-5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
          }
        }
      }
      if (still) break;
    }
    if (processed == 0) throw std::runtime_error("input contains no decodable frames");
    if (!output.empty() && !cv::imwrite(output, annotated))
      throw std::runtime_error("cannot write output: " + output);
    std::cout << "processed=" << processed << " positive_frames=" << positive_frames
              << " total_detections=" << total_detections
              << " mean_inference_ms=" << elapsed_ms / processed << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rune_detect: " << error.what() << '\n';
    return 1;
  }
}
