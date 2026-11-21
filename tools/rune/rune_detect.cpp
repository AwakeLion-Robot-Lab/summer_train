#include "rune/f1-observation/keypoint_detector.hpp"
#include "rune/debug_overlay.hpp"
#include "rune/f1-observation/image_refiner.hpp"
#include "runtime/auto_aim_config.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>
#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Milliseconds = std::chrono::duration<double, std::milli>;

void printTiming(const char* stage, std::vector<double>& samples) {
  std::sort(samples.begin(), samples.end());
  const auto index=static_cast<std::size_t>(std::ceil(0.95 * samples.size()))-1;
  const double mean=std::accumulate(samples.begin(), samples.end(), 0.0)/samples.size();
  std::cout << " mean_" << stage << "_ms=" << mean
            << " p95_" << stage << "_ms=" << samples[index];
}

nlohmann::json featureMeasurements(const std::vector<Rune::Keypoints>& detections) {
  auto result=nlohmann::json::array();
  const auto point=[](cv::Point2f p) { return nlohmann::json::array({p.x,p.y}); };
  for (const auto& detection : detections) {
    nlohmann::json entry{{"class",detection.class_id},{"ellipse",nullptr},{"center",nullptr},
                         {"gaps",nlohmann::json::array()},{"fan",nlohmann::json::object()}};
    if (const auto center=detection.armorCenter()) entry["center"]=point(*center);
    if (const auto& measured=detection.image_contours) {
      if (const auto& ellipse=measured->target_ellipse)
        entry["ellipse"]={{"center",point(ellipse->center)},
                           {"size",{ellipse->size.width,ellipse->size.height}},
                           {"angle",ellipse->angle}};
      for (const auto& p : measured->gap_points)
        entry["gaps"].push_back(p ? point(*p) : nlohmann::json(nullptr));
      for (std::size_t i=0;i<measured->fan_points.size();++i)
        if (measured->fanPointValid(i))
          entry["fan"][std::to_string(measured->fanPointId(i))]=point(measured->fan_points[i]);
    }
    result.push_back(std::move(entry));
  }
  return result;
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path config{"config/auto_aim.yaml"};
    std::string input, output, features_output;
    std::string backend;
    Rune::Mode mode = Rune::Mode::Small;
    Rune::Color color = Rune::Color::Blue;
    int frames = 1;
    for (int arg = 1; arg < argc; ++arg) {
      const std::string flag = argv[arg];
      if (flag == "--help") {
        std::cout << "rune_detect --input IMAGE_OR_VIDEO [--mode small|big] "
                     "[--config AUTO_AIM_YAML] [--backend scut|openvino] [--enemy blue|red] "
                     "[--frames N] [--output IMAGE.png] [--features-output AUDIT.jsonl]\n";
        return 0;
      }
      if (arg + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      const std::string value = argv[++arg];
      if (flag == "--input") input = value;
      else if (flag == "--output") output = value;
      else if (flag == "--features-output") features_output = value;
      else if (flag == "--config") config = value;
      else if (flag == "--backend") {
        if (value != "openvino" && value != "scut")
          throw std::invalid_argument("--backend must be scut or openvino");
        backend = value;
      }
      else if (flag == "--enemy") {
        if (value == "blue") color = Rune::Color::Blue;
        else if (value == "red") color = Rune::Color::Red;
        else throw std::invalid_argument("--enemy must be blue or red");
      }
      else if (flag == "--frames") frames = std::stoi(value);
      else if (flag == "--mode") {
        if (value == "small") mode = Rune::Mode::Small;
        else if (value == "big") mode = Rune::Mode::Big;
        else throw std::invalid_argument("--mode must be small or big");
      } else throw std::invalid_argument("unknown argument: " + flag);
    }
    if (input.empty() || frames < 1)
      throw std::invalid_argument("specify --input IMAGE_OR_VIDEO and a positive --frames count");
    auto detector_config = runtime::loadAutoAimConfig(config.string()).rune;
    if (!backend.empty()) {
      detector_config.detector_backend = backend;
      if (backend == "openvino") detector_config.detector = Rune::KeypointDetectorConfig::load(config);
      else {
        const auto rune = YAML::LoadFile(config.string())["rune"];
        detector_config.scut_detector = Rune::ScutContourConfig::fromYaml(
            rune["detector"]["scut"], rune["geometry"]);
      }
    }
    auto detector = runtime::makeRuneDetector(detector_config);
    cv::Mat image = cv::imread(input);
    const bool still = !image.empty();
    cv::VideoCapture video;
    if (!still && !video.open(input)) throw std::runtime_error("cannot open input: " + input);
    Rune::ImageRefiner refiner(Rune::ImageRefinerConfig::load(config));
    std::ofstream feature_log;
    if (!features_output.empty()) {
      const auto parent=std::filesystem::path(features_output).parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent);
      feature_log.open(features_output);
      if (!feature_log) throw std::runtime_error("cannot write feature audit: "+features_output);
    }
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
    std::vector<double> detection_ms, refinement_ms;
    while (processed < frames) {
      if (!still && !video.read(image)) break;
      const auto start = std::chrono::steady_clock::now();
      Rune::Frame frame;
      frame.bgr=image; frame.mode=mode; frame.color=color;
      double seconds=processed/std::max(1.0,fps);
      if (has_timestamps) {
        double w,x,y,z;
        if (!(poses>>seconds>>w>>x>>y>>z)) throw std::runtime_error("invalid or short pose timestamp file");
      }
      if (!std::isfinite(seconds) || seconds<=previous_seconds)
        throw std::runtime_error("video timestamps must increase");
      previous_seconds=seconds;
      frame.captured_at=epoch+std::chrono::duration_cast<Rune::TimePoint::duration>(std::chrono::duration<double>(seconds));
      const auto detection_start=std::chrono::steady_clock::now();
      frame.detections=detector->detect(image,mode,color);
      const auto detection_end=std::chrono::steady_clock::now();
      const auto raw_features=feature_log.is_open() ? featureMeasurements(frame.detections) : nlohmann::json{};
      const auto refinement_start=std::chrono::steady_clock::now();
      refiner.update(frame);
      const auto refinement_end=std::chrono::steady_clock::now();
      const double detection_sample=Milliseconds(detection_end-detection_start).count();
      const double refinement_sample=Milliseconds(refinement_end-refinement_start).count();
      detection_ms.push_back(detection_sample);
      refinement_ms.push_back(refinement_sample);
      if (feature_log.is_open()) {
        feature_log << nlohmann::json{{"frame",processed},{"t",seconds},{"raw",raw_features},
                                     {"stable",featureMeasurements(frame.detections)},
                                     {"detect_ms",detection_sample},{"refine_ms",refinement_sample}}.dump() << '\n';
        if (!feature_log) throw std::runtime_error("failed to write feature audit");
      }
      const auto& detections=frame.detections;
      elapsed_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      ++processed;
      positive_frames += !detections.empty();
      total_detections += detections.size();
      std::cout << "frame=" << processed << " detections=" << detections.size() << '\n';
      if (const auto diagnostics = detector->lastDiagnostics())
        std::cout << "  " << Rune::formatDetectionDiagnostics(*diagnostics) << '\n';
      const auto refinement=refiner.diagnostics();
      std::cout << "  image_refined_r=" << refinement.refined_r
                << " image_refined_targets=" << refinement.refined_targets
                << " ring_motion_reject=" << refinement.rejected_target_rings << '\n';
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
        if (const auto diagnostics = detector->lastDiagnostics())
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
    std::cout << "timing";
    printTiming("detect", detection_ms);
    printTiming("refine", refinement_ms);
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rune_detect: " << error.what() << '\n';
    return 1;
  }
}
