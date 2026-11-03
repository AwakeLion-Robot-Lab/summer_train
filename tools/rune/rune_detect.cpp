#include "rune/f1-observation/keypoint_detector.hpp"
#include "rune/debug_overlay.hpp"
#include "rune/f1-observation/image_refiner.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    std::filesystem::path config{"config/auto_aim.yaml"};
    std::string input, output;
    Rune::Mode mode = Rune::Mode::Small;
    int frames = 1;
    for (int arg = 1; arg < argc; ++arg) {
      const std::string flag = argv[arg];
      if (flag == "--help") {
        std::cout << "rune_detect --input IMAGE_OR_VIDEO [--mode small|big] "
                     "[--config AUTO_AIM_YAML] [--frames N] [--output IMAGE.png]\n";
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
    Rune::ImageRefiner refiner(Rune::ImageRefinerConfig::load(config));
    auto pose_path=std::filesystem::path(input);
    pose_path.replace_extension(".txt");
    std::ifstream poses;
    if (!still) poses.open(pose_path);
    const bool has_timestamps=poses.is_open();
    const double fps=still ? 1.0 : video.get(cv::CAP_PROP_FPS);
    if (!still && !has_timestamps && (!std::isfinite(fps) || fps<=0))
      throw std::runtime_error("video requires valid FPS or companion timestamps");
    const auto epoch=std::chrono::steady_clock::now();
    double previous_seconds=-1;
    cv::Mat annotated;
    int processed = 0, positive_frames = 0;
    std::size_t total_detections = 0;
    double elapsed_ms = 0;
    while (processed < frames) {
      if (!still && !video.read(image)) break;
      const auto start = std::chrono::steady_clock::now();
      Rune::Frame frame;
      frame.bgr=image; frame.mode=mode;
      double seconds=processed/std::max(1.0,fps);
      if (has_timestamps) {
        double w,x,y,z;
        if (!(poses>>seconds>>w>>x>>y>>z)) throw std::runtime_error("invalid or short pose timestamp file");
      }
      if (!std::isfinite(seconds) || seconds<=previous_seconds)
        throw std::runtime_error("video timestamps must increase");
      previous_seconds=seconds;
      frame.captured_at=epoch+std::chrono::duration_cast<Rune::TimePoint::duration>(std::chrono::duration<double>(seconds));
      frame.detections=detector.detect(image,mode);
      refiner.update(frame);
      const auto& detections=frame.detections;
      elapsed_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      ++processed;
      positive_frames += !detections.empty();
      total_detections += detections.size();
      std::cout << "frame=" << processed << " detections=" << detections.size() << '\n';
      if (const auto diagnostics = detector.lastDiagnostics())
        std::cout << "  " << Rune::formatDetectionDiagnostics(*diagnostics) << '\n';
      const auto refinement=refiner.diagnostics();
      std::cout << "  image_refined_r=" << refinement.refined_r
                << " image_refined_targets=" << refinement.refined_targets << '\n';
      if (!output.empty()) annotated = image.clone();
      for (const auto& detection : detections) {
        const std::array<cv::Point2f, 5> points{detection.top, detection.left,
          detection.right, detection.bottom, detection.center_r};
        std::cout << "  class=" << detection.class_id << " confidence=" << detection.confidence;
        for (const auto& point : points) std::cout << " (" << point.x << ',' << point.y << ')';
        std::cout << " kpt_conf=";
        for (const auto score : detection.point_confidences) std::cout << score << ',';
        std::cout << " valid=";
        for (const auto valid : detection.point_valid) std::cout << valid;
        std::cout << '\n';
      }
      if (!output.empty()) {
        Rune::drawDetectionOverlay(annotated, detections, mode);
        if (const auto diagnostics = detector.lastDiagnostics())
          Rune::drawDetectionDiagnostics(annotated, *diagnostics);
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
