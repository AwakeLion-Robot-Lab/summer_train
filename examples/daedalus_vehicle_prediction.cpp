#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/simulator/daedalus_client.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l3_estimation/armor/vehicle_model.hpp"
#include "l4_planning/armor/predictor.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/logger.hpp"
#include "runtime/auto_aim_config.hpp"

#include <Eigen/Geometry>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace VM = L3Estimation::VehicleModel;

std::atomic<bool> running{true};

void stopOnSignal(int)
{
  running = false;
}

struct Options
{
  L1Sensor::DaedalusPaths paths;
  std::string config_path{"config/auto_aim.yaml"};
  std::string save_path;
  L2Perception::ArmorColor enemy_color{L2Perception::ArmorColor::Unknown};
  std::size_t frame_limit{0};
  int connect_timeout_ms{5000};
  double predict_ms{100.0};
  double display_scale{0.7};
  bool headless{false};
};

std::string valueAfter(std::string_view argument, std::string_view prefix)
{
  if (!argument.starts_with(prefix)) {
    return {};
  }
  return std::string{argument.substr(prefix.size())};
}

L2Perception::ArmorColor parseEnemyColor(std::string_view value)
{
  if (value == "red") {
    return L2Perception::ArmorColor::Red;
  }
  if (value == "blue") {
    return L2Perception::ArmorColor::Blue;
  }
  if (value == "any") {
    return L2Perception::ArmorColor::Unknown;
  }
  throw std::invalid_argument("--enemy must be red, blue, or any");
}

Options parseOptions(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--help" || argument == "-h") {
      std::cout
        << "Usage: daedalus_vehicle_prediction [options]\n"
        << "  --frames=N                stop after N frames (0 means forever)\n"
        << "  --predict-ms=MS           whole-vehicle prediction horizon\n"
        << "  --enemy=red|blue|any      target color (default: any)\n"
        << "  --display-scale=S         display resize ratio (default: 0.7)\n"
        << "  --config=PATH             auto-aim YAML path\n"
        << "  --save=PATH               save the last annotated frame\n"
        << "  --headless                do not create an OpenCV window\n"
        << "  --connect-timeout-ms=MS   wait for simulator startup\n"
        << "  --metadata=PATH           override metadata mmap file\n"
        << "  --image-pool=PATH         override image-pool mmap file\n\n"
        << "cyan=detection, green=current vehicle, orange=future vehicle.\n"
        << "This viewer never sends a gimbal or fire command.\n";
      std::exit(0);
    }

    if (argument == "--headless") {
      options.headless = true;
    } else if (const auto value = valueAfter(argument, "--frames=");
               !value.empty()) {
      options.frame_limit = std::stoull(value);
    } else if (const auto value = valueAfter(argument, "--predict-ms=");
               !value.empty()) {
      options.predict_ms = std::stod(value);
    } else if (const auto value = valueAfter(argument, "--enemy=");
               !value.empty()) {
      options.enemy_color = parseEnemyColor(value);
    } else if (const auto value = valueAfter(argument, "--display-scale=");
               !value.empty()) {
      options.display_scale = std::stod(value);
    } else if (const auto value = valueAfter(argument, "--config=");
               !value.empty()) {
      options.config_path = value;
    } else if (const auto value = valueAfter(argument, "--save=");
               !value.empty()) {
      options.save_path = value;
    } else if (const auto value = valueAfter(argument, "--connect-timeout-ms=");
               !value.empty()) {
      options.connect_timeout_ms = std::stoi(value);
    } else if (const auto value = valueAfter(argument, "--metadata=");
               !value.empty()) {
      options.paths.metadata = value;
    } else if (const auto value = valueAfter(argument, "--image-pool=");
               !value.empty()) {
      options.paths.image_pool = value;
    } else {
      throw std::invalid_argument(
        "unknown or incomplete option: " + std::string{argument});
    }
  }

  if (options.connect_timeout_ms < 0) {
    throw std::invalid_argument("--connect-timeout-ms must not be negative");
  }
  if (!std::isfinite(options.predict_ms) || options.predict_ms < 0.0) {
    throw std::invalid_argument("--predict-ms must be finite and non-negative");
  }
  if (!std::isfinite(options.display_scale) || options.display_scale <= 0.0 ||
      options.display_scale > 2.0) {
    throw std::invalid_argument("--display-scale must be in (0, 2]");
  }
  if (options.headless && options.frame_limit == 0) {
    throw std::invalid_argument("--headless requires a non-zero --frames value");
  }
  return options;
}

bool connectWithRetry(
  L1Sensor::DaedalusClient& client,
  std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    if (client.connect() && client.isSimulatorAlive()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

L1Sensor::CameraCalibration makeCalibration(
  const L1Sensor::DaedalusFrame& frame)
{
  const auto& camera = frame.camera_info;
  if (!camera.valid() ||
      camera.width != static_cast<std::uint32_t>(frame.image_bgr.cols) ||
      camera.height != static_cast<std::uint32_t>(frame.image_bgr.rows)) {
    throw std::runtime_error("Daedalus camera metadata is missing or inconsistent");
  }

  L1Sensor::CameraCalibration calibration;
  calibration.image_size = {
    static_cast<int>(camera.width), static_cast<int>(camera.height)};
  calibration.camera_matrix = (
    cv::Mat_<double>(3, 3) <<
      camera.fx, 0.0, camera.cx,
      0.0, camera.fy, camera.cy,
      0.0, 0.0, 1.0);
  calibration.distortion_coefficients = cv::Mat(1, 5, CV_64FC1);
  for (std::size_t index = 0; index < camera.distortion.size(); ++index) {
    calibration.distortion_coefficients.at<double>(0, static_cast<int>(index)) =
      camera.distortion[index];
  }

  // Daedalus publishes the Camera slot translation in barrel coordinates.
  // Optical axes are z-forward/x-right/y-down, while newvision's barrel axes
  // are x-forward/y-left/z-up, hence p_barrel = [z, -x, -y].
  Eigen::Isometry3d T_barrel_camera = Eigen::Isometry3d::Identity();
  T_barrel_camera.linear() <<
     0.0,  0.0,  1.0,
    -1.0,  0.0,  0.0,
     0.0, -1.0,  0.0;
  const auto& camera_pose = frame.pose(L1Sensor::DaedalusPoseKind::Camera);
  T_barrel_camera.translation() = Eigen::Vector3d{
    camera_pose.position[0], camera_pose.position[1], camera_pose.position[2]};
  if (!T_barrel_camera.matrix().allFinite()) {
    throw std::runtime_error("Daedalus camera extrinsics contain a non-finite value");
  }
  calibration.T_barrel_camera = T_barrel_camera;
  return calibration;
}

std::optional<Eigen::Quaterniond> barrelPose(
  const L1Sensor::DaedalusFrame& frame)
{
  const auto& source = frame.pose(L1Sensor::DaedalusPoseKind::Gimbal).quaternion;
  Eigen::Quaterniond pose{source[0], source[1], source[2], source[3]};
  if (!pose.coeffs().allFinite() || pose.squaredNorm() <= 1e-12) {
    return std::nullopt;
  }
  pose.normalize();
  return pose;
}

std::string_view trackStateName(L3Estimation::TrackState state) noexcept
{
  switch (state) {
    case L3Estimation::TrackState::Lost:      return "lost";
    case L3Estimation::TrackState::Detecting: return "detecting";
    case L3Estimation::TrackState::Tracking:  return "tracking";
    case L3Estimation::TrackState::TempLost:  return "temp-lost";
  }
  return "unknown";
}

std::string_view armorName(L3Estimation::ArmorName name) noexcept
{
  switch (name) {
    case L3Estimation::ArmorName::Guard:      return "guard";
    case L3Estimation::ArmorName::Hero:       return "hero";
    case L3Estimation::ArmorName::Engineer:   return "engineer";
    case L3Estimation::ArmorName::Infantry3:  return "infantry-3";
    case L3Estimation::ArmorName::Infantry4:  return "infantry-4";
    case L3Estimation::ArmorName::Infantry5:  return "infantry-5";
    case L3Estimation::ArmorName::Outpost:    return "outpost";
    case L3Estimation::ArmorName::BaseSmall:  return "base-small";
    case L3Estimation::ArmorName::BaseLarge:  return "base-large";
    case L3Estimation::ArmorName::Unknown:    break;
  }
  return "-";
}

void drawDetections(
  cv::Mat& image, const std::vector<L2Perception::Armor>& detections)
{
  for (const auto& armor : detections) {
    for (std::size_t index = 0; index < armor.corners.size(); ++index) {
      cv::line(
        image, L6Telemetry::toPixel(armor.corners[index]),
        L6Telemetry::toPixel(
          armor.corners[(index + 1) % armor.corners.size()]),
        {255, 255, 0}, 1, cv::LINE_AA);
    }
  }
}

void drawPredictionLabels(
  cv::Mat& image,
  const std::vector<Eigen::Vector4d>& predicted_poses,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel,
  double predict_ms)
{
  for (std::size_t index = 0; index < predicted_poses.size(); ++index) {
    const auto pixel = L6Telemetry::projectWorldPoint(
      predicted_poses[index].head<3>(), calibration, q_world_barrel);
    if (!pixel) {
      continue;
    }
    L6Telemetry::drawOutlinedText(
      image,
      "+" + std::to_string(static_cast<int>(std::lround(predict_ms))) +
        "ms #" + std::to_string(index),
      L6Telemetry::toPixel(*pixel) +
        cv::Point{5, -5 - static_cast<int>(index % 2) * 18},
      {0, 165, 255}, 0.45);
  }
}

void drawCenterMotion(
  cv::Mat& image,
  const L3Estimation::EskfTarget& current,
  const L3Estimation::EskfTarget& predicted,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Quaterniond& q_world_barrel)
{
  const auto& current_state = current.rawState();
  const auto& predicted_state = predicted.rawState();
  const Eigen::Vector3d current_center{
    current_state[VM::idx::CX], current_state[VM::idx::CY],
    current_state[VM::idx::CZ]};
  const Eigen::Vector3d predicted_center{
    predicted_state[VM::idx::CX], predicted_state[VM::idx::CY],
    predicted_state[VM::idx::CZ]};
  const auto current_pixel = L6Telemetry::projectWorldPoint(
    current_center, calibration, q_world_barrel);
  const auto predicted_pixel = L6Telemetry::projectWorldPoint(
    predicted_center, calibration, q_world_barrel);
  if (!current_pixel || !predicted_pixel) {
    return;
  }

  const cv::Point start = L6Telemetry::toPixel(*current_pixel);
  const cv::Point finish = L6Telemetry::toPixel(*predicted_pixel);
  cv::circle(image, start, 6, {0, 255, 0}, cv::FILLED, cv::LINE_AA);
  cv::circle(image, finish, 6, {0, 165, 255}, cv::FILLED, cv::LINE_AA);
  if (cv::norm(finish - start) > 2.0) {
    cv::arrowedLine(
      image, start, finish, {0, 165, 255}, 2, cv::LINE_AA, 0, 0.25);
  }
}

void drawStatus(
  cv::Mat& image,
  std::uint64_t sequence,
  L3Estimation::TrackState state,
  const std::optional<L3Estimation::EskfTarget>& target,
  std::size_t detection_count,
  int match_count,
  double predict_ms)
{
  std::ostringstream first;
  first << "frame=" << sequence << " | ieskf=" << trackStateName(state)
        << " | target=" << (target ? armorName(target->name) : "-")
        << " | detections=" << detection_count << " | matches=" << match_count;
  L6Telemetry::drawOutlinedText(
    image, first.str(), {12, 30}, {255, 255, 255}, 0.62);

  std::ostringstream second;
  second << "cyan=detection | green=current vehicle | orange=+"
         << static_cast<int>(std::lround(predict_ms)) << "ms prediction";
  L6Telemetry::drawOutlinedText(
    image, second.str(), {12, 60}, {0, 220, 255}, 0.55);

  if (!target) {
    return;
  }
  const auto& x = target->rawState();
  std::ostringstream motion;
  motion.setf(std::ios::fixed);
  motion.precision(2);
  motion << "center=(" << x[VM::idx::CX] << ',' << x[VM::idx::CY] << ','
         << x[VM::idx::CZ] << ")m | velocity=(" << x[VM::idx::VCX] << ','
         << x[VM::idx::VCY] << ',' << x[VM::idx::VCZ]
         << ")m/s | yaw_rate=" << x[VM::idx::VYAW] << "rad/s";
  L6Telemetry::drawOutlinedText(
    image, motion.str(), {12, 90}, {0, 255, 0}, 0.52);
}

L2Perception::ArmorDetector makeDetector(const runtime::AutoAimConfig& config)
{
  auto backend = L2Perception::makeInferenceBackend(config.inference_backend);
  backend->load(config.inference);
  if (!backend->ready()) {
    throw std::runtime_error("configured inference backend is not ready");
  }
  L2Perception::NumberClassifier classifier;
  classifier.load(config.number_classifier);
  return L2Perception::ArmorDetector(
    std::move(backend), std::move(classifier), config.light_decoder,
    config.light_matcher);
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const Options options = parseOptions(argc, argv);
    std::signal(SIGINT, stopOnSignal);
    std::signal(SIGTERM, stopOnSignal);
    L6Telemetry::initLogger();

    L1Sensor::DaedalusClient client(options.paths);
    if (!connectWithRetry(
          client, std::chrono::milliseconds{options.connect_timeout_ms})) {
      const std::string error = client.lastError();
      throw std::runtime_error(
        "cannot connect to a live Daedalus simulator: " +
        (error.empty() ? std::string{"shared-memory heartbeat is stale"} : error));
    }

    L1Sensor::DaedalusFrame first_frame;
    if (!client.readFrame(first_frame, std::chrono::milliseconds{2000})) {
      throw std::runtime_error(
        "cannot read the first Daedalus frame: " + client.lastError());
    }
    const L1Sensor::CameraCalibration calibration = makeCalibration(first_frame);
    const runtime::AutoAimConfig config =
      runtime::loadAutoAimConfig(options.config_path);
    L2Perception::ArmorDetector detector = makeDetector(config);
    L3Estimation::EskfTracker tracker(
      calibration, config.armor, config.ieskf_tracker, config.ieskf_target);
    L3Estimation::PnpSolver overlay_solver(calibration, config.armor);
    if (!tracker.ready() || !overlay_solver.ready()) {
      throw std::runtime_error("Daedalus calibration was rejected by L3");
    }
    L4Planning::Predictor predictor;

    std::cout << "Daedalus whole-vehicle prediction viewer ready | camera="
              << calibration.image_size.width << 'x'
              << calibration.image_size.height << " | horizon="
              << options.predict_ms << " ms | visualization-only\n";

    if (!options.headless) {
      cv::namedWindow("Daedalus vehicle prediction", cv::WINDOW_NORMAL);
    }

    const double predict_seconds = options.predict_ms * 1e-3;
    std::size_t processed = 0;
    cv::Mat last_annotated;
    std::optional<L1Sensor::DaedalusFrame> pending{std::move(first_frame)};
    while (running &&
           (options.frame_limit == 0 || processed < options.frame_limit)) {
      L1Sensor::DaedalusFrame frame;
      if (pending) {
        frame = std::move(*pending);
        pending.reset();
      } else if (!client.readFrame(frame, std::chrono::milliseconds{1000})) {
        const std::string error = client.lastError();
        if (!error.empty()) {
          throw std::runtime_error("Daedalus frame read failed: " + error);
        }
        if (!client.isSimulatorAlive()) {
          throw std::runtime_error("Daedalus heartbeat stopped");
        }
        continue;
      }

      const auto q_world_barrel = barrelPose(frame);
      if (!q_world_barrel) {
        throw std::runtime_error("Daedalus published an invalid barrel quaternion");
      }

      const auto light_roi = tracker.lightDetectionRoi(
        q_world_barrel, frame.capture_time, frame.image_bgr.size());
      const cv::Rect net_roi = tracker.netFocusRoi(
        q_world_barrel, frame.capture_time, frame.image_bgr.size(),
        detector.networkAspectRatio());
      L2Perception::ArmorFrame perception = detector.detectFrame(
        frame.image_bgr, light_roi, net_roi, options.enemy_color);
      const std::vector<L2Perception::Armor> all_detections = perception.armors;
      if (options.enemy_color != L2Perception::ArmorColor::Unknown) {
        std::erase_if(
          perception.armors, [&options](const L2Perception::Armor& armor) {
            return armor.color != options.enemy_color;
          });
      }

      const auto target = tracker.track(
        perception.armors, perception.lights, q_world_barrel,
        frame.capture_time);
      overlay_solver.set_R_world_barrel(q_world_barrel);
      drawDetections(frame.image_bgr, all_detections);

      if (target) {
        const auto current_poses = target->armor_xyza_list();
        const auto predicted = predictor.predict(*target, predict_seconds);
        const auto predicted_poses = predicted.armor_xyza_list();
        const auto type = L3Estimation::armorTypeOf(target->name)
                            .value_or(L3Estimation::ArmorType::Small);
        L6Telemetry::drawVehicle(
          frame.image_bgr, current_poses, type, target->name, overlay_solver,
          {0, 255, 0}, 3);
        L6Telemetry::drawVehicle(
          frame.image_bgr, predicted_poses, type, target->name, overlay_solver,
          {0, 165, 255}, 1);
        drawCenterMotion(
          frame.image_bgr, *target, predicted, calibration, *q_world_barrel);
        drawPredictionLabels(
          frame.image_bgr, predicted_poses, calibration, *q_world_barrel,
          options.predict_ms);
      }

      drawStatus(
        frame.image_bgr, frame.sequence, tracker.state(), target,
        all_detections.size(), tracker.lastMatchCount(), options.predict_ms);
      last_annotated = frame.image_bgr;
      ++processed;

      if (processed == 1 || processed % 30 == 0) {
        std::cout << "frame=" << frame.sequence
                  << " state=" << trackStateName(tracker.state())
                  << " detections=" << all_detections.size()
                  << " matches=" << tracker.lastMatchCount()
                  << " target=" << (target ? armorName(target->name) : "-")
                  << '\n';
      }

      if (!options.headless) {
        cv::Mat display;
        cv::resize(
          frame.image_bgr, display, {}, options.display_scale,
          options.display_scale, cv::INTER_AREA);
        cv::imshow("Daedalus vehicle prediction", display);
        const int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
          break;
        }
      }
    }

    if (!options.save_path.empty()) {
      if (last_annotated.empty() ||
          !cv::imwrite(options.save_path, last_annotated)) {
        throw std::runtime_error(
          "failed to save annotated frame to " + options.save_path);
      }
      std::cout << "Saved annotated frame to " << options.save_path << '\n';
    }
    if (!options.headless) {
      cv::destroyWindow("Daedalus vehicle prediction");
    }
    L6Telemetry::flushLogger();
    std::cout << "Stopped after " << processed << " frame(s)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
