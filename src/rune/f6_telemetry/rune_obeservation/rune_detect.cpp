#include "rune/f1-observation/scut_contour_detector.hpp"
#include "rune/f6_telemetry/debug_overlay.hpp"
#include "rune/f1-observation/image_refiner.hpp"
#include "runtime/auto_aim_config.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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
    entry["center_r"]=point(detection.center_r);
    entry["anchors"]={point(detection.top),point(detection.left),point(detection.right),
                      point(detection.bottom),point(detection.center_r)};
    entry["anchor_valid"]=detection.point_valid;
    entry["confidence"]=detection.confidence;
    entry["variance_scale"]={detection.center_r_variance_scale,detection.target_variance_scale};
    if (const auto center=detection.armorCenter()) entry["center"]=point(*center);
    if (const auto& measured=detection.image_contours) {
      if (const auto& ellipse=measured->target_ellipse)
        entry["ellipse"]={{"center",point(ellipse->center)},
                           {"size",{ellipse->size.width,ellipse->size.height}},
                           {"angle",ellipse->angle}};
      for (const auto& p : measured->gap_points)
        entry["gaps"].push_back(p ? point(*p) : nlohmann::json(nullptr));
      for (std::size_t i=0; i<measured->gap_display_predictions.size(); ++i)
        if (const auto& predicted=measured->gap_display_predictions[i])
          entry["gap_display_predictions"][std::to_string(i+1)]=point(*predicted);
      for (std::size_t i=0;i<measured->fan_points.size();++i) {
        const auto id = std::to_string(measured->fanPointId(i));
        if (measured->fanPointValid(i))
          entry["fan"][id]=point(measured->fan_points[i]);
      }
      for (std::size_t id=0; id<measured->fan_display_predictions.size(); ++id)
        if (const auto& predicted=measured->fan_display_predictions[id])
          entry["fan_display_predictions"][std::to_string(id)]=point(*predicted);
      if (const auto& outline=measured->inactive_arm_outline) {
        auto corners=nlohmann::json::array(),edges=nlohmann::json::array();
        for (const auto& corner:outline->corners) corners.push_back(point(corner));
        for (const auto& edge:outline->measured_edges)
          edges.push_back(edge ? nlohmann::json::array({point((*edge)[0]),point((*edge)[1])})
                               : nlohmann::json(nullptr));
        entry["inactive_arm_outline"]={{"corners",std::move(corners)},
                                       {"measured_edges",std::move(edges)}};
      }
    }
    result.push_back(std::move(entry));
  }
  return result;
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path config{"config/auto_aim.yaml"};
    std::string input, output, features_output, ring_reobservation, arm_refinement;
    std::string registration_backend, replay_clock="unpaced";
    Rune::Mode mode = Rune::Mode::Small;
    Rune::Color color = Rune::Color::Blue;
    int frames = 1;
    bool quiet = false;
    for (int arg = 1; arg < argc; ++arg) {
      const std::string flag = argv[arg];
      if (flag == "--help") {
        std::cout << "rune_detect --input IMAGE_OR_VIDEO [--mode small|big] "
                     "[--config AUTO_AIM_YAML] [--enemy blue|red] "
                     "[--frames N] [--output IMAGE.png] [--features-output AUDIT.jsonl] "
                     "[--registration-backend cpu|opencl] [--replay-clock realtime|unpaced] "
                     "[--ring-reobservation on|off] [--arm-refinement on|off] [--quiet]\n";
        return 0;
      }
      if (flag == "--quiet") {
        quiet = true;
        continue;
      }
      if (arg + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      const std::string value = argv[++arg];
      if (flag == "--input") input = value;
      else if (flag == "--output") output = value;
      else if (flag == "--features-output") features_output = value;
      else if (flag == "--config") config = value;
      else if (flag == "--registration-backend") registration_backend = value;
      else if (flag == "--replay-clock") replay_clock = value;
      else if (flag == "--ring-reobservation") ring_reobservation = value;
      else if (flag == "--arm-refinement") arm_refinement = value;
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
    if (replay_clock != "realtime" && replay_clock != "unpaced")
      throw std::invalid_argument("--replay-clock must be realtime or unpaced");
    const auto rune_config = runtime::loadAutoAimConfig(config.string()).rune;
    Rune::ScutContourDetector detector(rune_config.scut_detector, rune_config.gimbal_only);
    cv::Mat image = cv::imread(input);
    const bool still = !image.empty();
    cv::VideoCapture video;
    if (!still && !video.open(input)) throw std::runtime_error("cannot open input: " + input);
    auto refiner_config = rune_config.image_refiner;
    if (!arm_refinement.empty()) {
      if (arm_refinement != "on" && arm_refinement != "off")
        throw std::invalid_argument("--arm-refinement must be on or off");
      refiner_config.refine_inactive_arm_edges = arm_refinement == "on";
    }
    if (!ring_reobservation.empty()) {
      if (ring_reobservation != "on" && ring_reobservation != "off")
        throw std::invalid_argument("--ring-reobservation must be on or off");
      refiner_config.reobserve_target_rings = ring_reobservation == "on";
    }
    if (!registration_backend.empty()) {
      if (registration_backend != "cpu" && registration_backend != "opencl")
        throw std::invalid_argument("--registration-backend must be cpu or opencl");
      refiner_config.use_opencl = registration_backend == "opencl";
    }
    Rune::ImageRefiner refiner(refiner_config);
    if (!quiet && refiner_config.enabled)
      std::cout << "image_registration_device=" << refiner.registrationDevice() << '\n';
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
    std::optional<double> first_seconds;
    std::optional<Rune::Frame> last_frame;
    std::optional<Rune::DetectorDiagnostics> last_diagnostics;
    int processed = 0, positive_frames = 0;
    std::size_t total_detections = 0;
    double elapsed_ms = 0;
    std::vector<double> detection_ms, refinement_ms;
    while (processed < frames) {
      if (!still && !video.read(image)) break;
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
      if (!first_seconds) first_seconds=seconds;
      frame.captured_at=epoch+std::chrono::duration_cast<Rune::TimePoint::duration>(
          std::chrono::duration<double>(seconds-*first_seconds));
      if (replay_clock == "realtime") std::this_thread::sleep_until(frame.captured_at);
      const auto start=std::chrono::steady_clock::now();
      const double replay_lag_ms=Milliseconds(start-frame.captured_at).count();
      const auto detection_start=std::chrono::steady_clock::now();
      frame.detections=detector.detect(frame.bgr, frame.mode, frame.color);
      const auto detection_end=std::chrono::steady_clock::now();
      const auto raw_features=feature_log.is_open() ? featureMeasurements(frame.detections) : nlohmann::json{};
      const auto refinement_start=std::chrono::steady_clock::now();
      refiner.update(frame);
      const auto refinement_end=std::chrono::steady_clock::now();
      const double detection_sample=Milliseconds(detection_end-detection_start).count();
      const double refinement_sample=Milliseconds(refinement_end-refinement_start).count();
      const auto refinement=refiner.diagnostics();
      detection_ms.push_back(detection_sample);
      refinement_ms.push_back(refinement_sample);
      const auto diagnostics = detector.lastDiagnostics();
      if (feature_log.is_open()) {
        nlohmann::json audit{{"frame",processed},{"t",seconds},{"raw",raw_features},
                                     {"stable",featureMeasurements(frame.detections)},
                                     {"detect_ms",detection_sample},{"refine_ms",refinement_sample},
                                     {"ring_reobservation_ms",refinement.ring_reobservation_ms},
                                     {"inactive_arm_ms",refinement.inactive_arm_ms},
                                     {"gpu_ecc_batches",refinement.gpu_ecc_batches},
                                     {"gpu_ecc_accepted",refinement.gpu_ecc_accepted},
                                     {"image_size",{image.cols,image.rows}},
                                     {"backend","scut"},
                                     {"replay_clock",replay_clock},{"replay_lag_ms",replay_lag_ms}};
        if (!frame.reobserved_target_rings.empty()) {
          auto rings=nlohmann::json::array();
          for (const auto& ring : frame.reobserved_target_rings) {
            const auto& center=ring.ellipse.center;
            rings.push_back({{"center",{center.x,center.y}},
                             {"source",ring.from_history ? "history_roi" :
                                       ring.from_shared_motion ? "shared_motion_roi" : "optical_flow_roi"},
                             {"arc_coverage",ring.arc_coverage},{"fit_rms_px",ring.fit_rms_px}});
          }
          audit["reobserved_rings"]=std::move(rings);
        }
        if (!frame.display_outline_predictions.empty()) {
          auto outlines=nlohmann::json::array();
          for (const auto& prediction : frame.display_outline_predictions) {
            const auto& e=prediction.target;
            outlines.push_back({{"center",{e.center.x,e.center.y}},
                                {"size",{e.size.width,e.size.height}},
                                {"class",prediction.class_id}});
          }
          audit["display_predictions"]=std::move(outlines);
        }
        feature_log << audit.dump() << '\n';
        if (!feature_log) throw std::runtime_error("failed to write feature audit");
      }
      const auto& detections=frame.detections;
      elapsed_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
      ++processed;
      positive_frames += !detections.empty();
      total_detections += detections.size();
      if (!quiet) {
        std::cout << "frame=" << processed << " detections=" << detections.size() << '\n';
        if (diagnostics)
          std::cout << "  " << Rune::formatDetectionDiagnostics(*diagnostics) << '\n';
        std::cout << "  image_refined_r=" << refinement.refined_r
                  << " hub_registrations=" << refinement.hub_registrations
                  << " image_refined_targets=" << refinement.refined_targets
                  << " ring_motion_reject=" << refinement.rejected_target_rings;
        if (refiner_config.use_opencl)
          std::cout << " gpu_ecc_batches=" << refinement.gpu_ecc_batches
                    << " gpu_ecc_accepted=" << refinement.gpu_ecc_accepted;
        std::cout << '\n';
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
      }
      if (!output.empty()) {
        // Retain the decoded pixels and prevent the next read from overwriting them.
        image.release();
        last_frame = std::move(frame);
        last_diagnostics = diagnostics;
      }
      if (still) break;
    }
    if (processed == 0) throw std::runtime_error("input contains no decodable frames");
    if (last_frame) {
      auto annotated = last_frame->bgr.clone();
      Rune::drawDetectionOverlay(annotated, *last_frame);
      if (last_diagnostics)
        Rune::drawDetectionDiagnostics(annotated, *last_diagnostics);
      if (!cv::imwrite(output, annotated))
        throw std::runtime_error("cannot write output: " + output);
    }
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
