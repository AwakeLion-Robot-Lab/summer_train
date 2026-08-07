#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l6_telemetry/l3_replay_json.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "l6_telemetry/udp_json_sender.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kRadToDeg = 180.0 / std::numbers::pi;

const char* kCommandLineKeys =
  "{help h usage ? | | 显示命令行帮助}"
  "{camera-config c | config/carmera_config.yaml | 相机标定 YAML}"
  "{l3-config | config/l3_config.yaml | L3 参数 YAML}"
  "{model-path m | model/armor_model/armor.xml | OpenVINO 装甲模型}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{mode | realtime | 回放模式：realtime 或 offline}"
  "{prediction-ms | 100 | 整车未来轮廓时长，0 仅关闭未来层}"
  "{start-index s | 0 | 起始帧下标}"
  "{end-index e | 0 | 结束帧下标，0 表示文件末尾}"
  "{robot-id | -1 | PlotJuggler 目标 ID，-1 自动锁定}"
  "{plotjuggler-host | 127.0.0.1 | PlotJuggler UDP IPv4 地址}"
  "{plotjuggler-port | 9870 | PlotJuggler UDP 端口}"
  "{no-geometry-constraints | | 关闭整车半径/夹角/一对一关联约束}"
  "{no-ippe-dual-candidates | | 关闭 IPPE 双候选，恢复单候选基线}"
  "{no-predicted-face-yaw-selection | | 关闭预测 face yaw 选解，恢复 yaw 误差选解}"
  "{show-armor-text | | 启动时显示装甲板详细文字}"
  "{no-gui | | 关闭 OpenCV 界面并禁用键盘控制}"
  "{no-plotjuggler | | 关闭 PlotJuggler UDP 输出}"
  "{@input-path | | record_capture 输出路径，不带扩展名}";

struct PoseRecord {
  double time_seconds = 0.0;
  Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

enum class PlaybackMode {
  Realtime,
  Offline,
};

enum class AdvanceAction {
  Timed,
  Step,
  Redraw,
  Quit,
};

std::filesystem::path recordBase(std::filesystem::path path)
{
  if (path.extension() == ".avi" || path.extension() == ".txt") {
    path.replace_extension();
  }
  return path;
}

std::vector<PoseRecord> loadPoseRecords(const std::filesystem::path& path)
{
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("pose file does not exist: " + path.string());
  }
  if (std::filesystem::file_size(path) == 0) {
    throw std::runtime_error("pose file is empty: " + path.string());
  }

  std::ifstream input{path};
  if (!input) {
    throw std::runtime_error("failed to open pose file: " + path.string());
  }

  std::vector<PoseRecord> records;
  double previous_time = -1.0;
  while (true) {
    PoseRecord record;
    double w = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(input >> record.time_seconds >> w >> x >> y >> z)) {
      if (input.eof()) {
        break;
      }
      throw std::runtime_error(
        "malformed pose record at line " + std::to_string(records.size() + 1));
    }

    record.orientation = Eigen::Quaterniond{w, x, y, z};
    if (!std::isfinite(record.time_seconds)
        || record.time_seconds < 0.0
        || (!records.empty() && record.time_seconds <= previous_time)
        || !record.orientation.coeffs().allFinite()
        || record.orientation.norm() <= 1e-9) {
      throw std::runtime_error(
        "invalid timestamp or quaternion at pose line "
        + std::to_string(records.size() + 1));
    }
    record.orientation.normalize();
    previous_time = record.time_seconds;
    records.push_back(record);
  }

  if (records.empty()) {
    throw std::runtime_error("pose file contains no records: " + path.string());
  }
  return records;
}

L1Sensor::CameraCalibration loadCalibration(
  const std::filesystem::path& config_path)
{
  const YAML::Node root = YAML::LoadFile(config_path.string());
  auto calibration = L1Sensor::loadCameraCalibration(
    root["calibration"], config_path.string() + ": calibration");
  if (!calibration.barrelExtrinsicsReady()) {
    throw std::runtime_error(
      "camera calibration is missing calibration.T_barrel_camera");
  }
  return calibration;
}

L2Perception::ArmorDetector makeDetector(
  const std::filesystem::path& model_path,
  const std::string& device)
{
  auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
  L2Perception::InferenceModelConfig config;
  config.model_path = model_path;
  config.device = device;
  config.model_color_order = L2Perception::ModelColorOrder::Rgb;
  config.normalization_divisor = 255.0F;
  backend->load(config);

  L2Perception::ArmorDetector detector{std::move(backend)};
  if (!detector.ready()) {
    throw std::runtime_error("armor detector is not ready");
  }
  return detector;
}

std::string armorClassName(int class_id)
{
  static const std::vector<std::string> names{
    "guard", "hero", "engineer", "infantry3", "infantry4",
    "infantry5", "outpost", "base-small", "base-large"};
  return class_id >= 0 && static_cast<std::size_t>(class_id) < names.size()
           ? names[static_cast<std::size_t>(class_id)]
           : "unknown";
}

std::string trackerStateName(L3Estimation::TrackerState state)
{
  switch (state) {
  case L3Estimation::TrackerState::Lost:
    return "lost";
  case L3Estimation::TrackerState::Detecting:
    return "detecting";
  case L3Estimation::TrackerState::Tracking:
    return "tracking";
  case L3Estimation::TrackerState::TemporaryLost:
    return "temporary-lost";
  }
  return "unknown";
}

cv::Scalar detectionColor(L2Perception::ArmorColor color)
{
  if (color == L2Perception::ArmorColor::Red) {
    return {0, 0, 255};
  }
  if (color == L2Perception::ArmorColor::Blue) {
    return {255, 0, 0};
  }
  return {0, 255, 255};
}

void drawText(
  cv::Mat& image,
  const std::string& text,
  cv::Point origin,
  const cv::Scalar& color,
  double scale = 0.48)
{
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX,
    scale, {0, 0, 0}, 3, cv::LINE_AA);
  cv::putText(
    image, text, origin, cv::FONT_HERSHEY_SIMPLEX,
    scale, color, 1, cv::LINE_AA);
}

std::optional<cv::Point2f> projectWorldPoint(
  const Eigen::Vector3d& point_world,
  const Eigen::Isometry3d& T_camera_world,
  const L1Sensor::CameraCalibration& calibration)
{
  const Eigen::Vector3d point_camera = T_camera_world * point_world;
  if (!point_camera.allFinite() || point_camera.z() <= 0.01) {
    return std::nullopt;
  }

  std::vector<cv::Point3d> points{{
    point_camera.x(), point_camera.y(), point_camera.z()}};
  std::vector<cv::Point2d> pixels;
  cv::projectPoints(
    points,
    cv::Vec3d{0.0, 0.0, 0.0},
    cv::Vec3d{0.0, 0.0, 0.0},
    calibration.camera_matrix,
    calibration.distortion_coefficients,
    pixels);
  if (pixels.size() != 1
      || !std::isfinite(pixels[0].x)
      || !std::isfinite(pixels[0].y)
      || std::abs(pixels[0].x) > 1e6
      || std::abs(pixels[0].y) > 1e6) {
    return std::nullopt;
  }
  return cv::Point2f{
    static_cast<float>(pixels[0].x),
    static_cast<float>(pixels[0].y)};
}

void drawVehicleModel(
  cv::Mat& image,
  const L3Estimation::TargetState& target,
  const L3Estimation::L3Config& config,
  const L1Sensor::CameraCalibration& calibration,
  const Eigen::Isometry3d& T_camera_world,
  double prediction_seconds,
  const cv::Scalar& color,
  int thickness,
  bool show_armor_text)
{
  const Eigen::Vector3d center =
    target.center + target.velocity * prediction_seconds;
  const double yaw = target.yaw + target.yaw_rate * prediction_seconds;
  if (!center.allFinite() || !std::isfinite(yaw)) {
    return;
  }

  const auto traits = L3Estimation::targetModelTraits(target.model);
  const auto& model_parameters = config.armor.parameters(target.model);
  const double armor_width =
    target.robot_id == static_cast<int>(L2Perception::ArmorClass::Hero)
      ? config.armor.dimensions.large_width
      : config.armor.dimensions.small_width;
  const double half_width = armor_width / 2.0;
  const double half_height = config.armor.dimensions.height / 2.0;
  const std::array<Eigen::Vector3d, 4> local_corners{{
    {0.0, half_width, half_height},
    {0.0, -half_width, half_height},
    {0.0, -half_width, -half_height},
    {0.0, half_width, -half_height}}};

  const auto center_pixel = projectWorldPoint(
    center, T_camera_world, calibration);
  std::vector<std::optional<cv::Point2f>> armor_centers;
  armor_centers.reserve(static_cast<std::size_t>(traits.armor_count));

  for (int face_id = 0; face_id < traits.armor_count; ++face_id) {
    const double angle = yaw
      + static_cast<double>(face_id) * traits.face_angle_interval_rad;
    const bool second_group =
      traits.uses_alternating_radius_and_height && face_id % 2 != 0;
    const double radius = target.radius
      + (second_group ? target.radius_offset : 0.0);
    const Eigen::Vector3d armor_center{
      center.x() - radius * std::cos(angle),
      center.y() - radius * std::sin(angle),
      center.z() + (second_group ? target.height_offset : 0.0)};
    armor_centers.push_back(projectWorldPoint(
      armor_center, T_camera_world, calibration));

    const Eigen::Matrix3d R_world_armor =
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix()
      * Eigen::AngleAxisd(
          model_parameters.pitch_rad,
          Eigen::Vector3d::UnitY()).toRotationMatrix();
    std::array<std::optional<cv::Point2f>, 4> corners;
    bool all_visible = true;
    for (std::size_t corner = 0; corner < corners.size(); ++corner) {
      corners[corner] = projectWorldPoint(
        armor_center + R_world_armor * local_corners[corner],
        T_camera_world,
        calibration);
      all_visible = all_visible && corners[corner].has_value();
    }
    if (all_visible) {
      for (std::size_t corner = 0; corner < corners.size(); ++corner) {
        cv::line(
          image,
          *corners[corner],
          *corners[(corner + 1) % corners.size()],
          color,
          thickness,
          cv::LINE_AA);
      }
    }
    if (show_armor_text && armor_centers.back()) {
      drawText(
        image,
        "P" + std::to_string(face_id),
        cv::Point{
          cvRound(armor_centers.back()->x) + 4,
          cvRound(armor_centers.back()->y) - 4},
        color,
        0.42);
    }
  }

  for (int face_id = 0; face_id < traits.armor_count; ++face_id) {
    const auto& begin = armor_centers[static_cast<std::size_t>(face_id)];
    const auto& end = armor_centers[
      static_cast<std::size_t>((face_id + 1) % traits.armor_count)];
    if (begin && end) {
      cv::line(image, *begin, *end, color, thickness, cv::LINE_AA);
    }
    if (center_pixel && begin) {
      cv::line(image, *center_pixel, *begin, color, 1, cv::LINE_AA);
    }
  }

  if (center_pixel) {
    cv::circle(image, *center_pixel, 4, color, cv::FILLED, cv::LINE_AA);
    std::ostringstream label;
    label << (prediction_seconds > 0.0 ? "+" : "EKF #")
          << (prediction_seconds > 0.0
                ? cvRound(prediction_seconds * 1000.0)
                : target.robot_id)
          << (prediction_seconds > 0.0 ? "ms" : "")
          << (target.updated_this_frame ? " U" : " P");
    drawText(
      image,
      label.str(),
      cv::Point{cvRound(center_pixel->x) + 7, cvRound(center_pixel->y) - 7},
      color,
      0.45);
  }
}

void drawFrame(
  cv::Mat& image,
  int frame_index,
  double time_seconds,
  double l2_ms,
  double l3_ms,
  const std::vector<L2Perception::ArmorDetection>& detections,
  const std::vector<L3Estimation::ArmorObservation>& observations,
  const std::vector<L3Estimation::TargetState>& targets,
  const std::optional<int>& selected_robot,
  const L1Sensor::CameraCalibration& calibration,
  const L3Estimation::L3Config& l3_parameters,
  const Eigen::Quaterniond& R_world_barrel,
  double prediction_seconds,
  bool geometry_constraints_enabled,
  PlaybackMode mode,
  std::size_t skipped_frames,
  double realtime_lag_ms,
  bool show_armor_text,
  bool paused)
{
  Eigen::Isometry3d T_barrel_world = Eigen::Isometry3d::Identity();
  T_barrel_world.linear() =
    R_world_barrel.normalized().toRotationMatrix().transpose();
  const Eigen::Isometry3d T_camera_world =
    calibration.T_barrel_camera->inverse() * T_barrel_world;
  for (const auto& target : targets) {
    if (prediction_seconds > 0.0) {
      drawVehicleModel(
        image, target, l3_parameters, calibration, T_camera_world,
        prediction_seconds, {255, 0, 255}, 1, false);
    }
    drawVehicleModel(
      image, target, l3_parameters, calibration, T_camera_world,
      0.0,
      target.updated_this_frame
        ? cv::Scalar{0, 255, 0}
        : cv::Scalar{0, 165, 255},
      2,
      show_armor_text);
  }

  for (std::size_t index = 0; index < detections.size(); ++index) {
    const auto& detection = detections[index];
    const cv::Scalar color = detectionColor(detection.color);
    for (std::size_t corner = 0; corner < detection.corners.size(); ++corner) {
      cv::line(
        image,
        detection.corners[corner],
        detection.corners[(corner + 1) % detection.corners.size()],
        color, 2, cv::LINE_AA);
    }

    if (show_armor_text) {
      const cv::Point label_origin{
        cvRound(detection.corners[0].x),
        std::max(20, cvRound(detection.corners[0].y) - 8)};
      std::ostringstream label;
      label << '#' << index << ' ' << armorClassName(detection.class_id)
            << " conf=" << std::fixed << std::setprecision(2)
            << detection.confidence;
      drawText(image, label.str(), label_origin, color);

      const auto observation = std::find_if(
        observations.begin(), observations.end(),
        [index](const auto& value) {
          return value.source_detection_index == index;
        });
      if (observation != observations.end()) {
        std::ostringstream position;
        position << "xyz=" << std::fixed << std::setprecision(2)
                 << observation->position_world.transpose();
        drawText(
          image, position.str(), label_origin + cv::Point{0, 20}, {0, 255, 0});

        const auto rpyText = [](const char* name, const Eigen::Vector3d& rpy) {
          std::ostringstream text;
          text << name << " rpy=" << std::fixed << std::setprecision(1)
               << (rpy * kRadToDeg).transpose() << "deg";
          return text.str();
        };
        drawText(
          image,
          rpyText("raw", observation->rpy_raw_world),
          label_origin + cv::Point{0, 40}, {0, 255, 0});
        drawText(
          image,
          rpyText("opt", observation->rpy_constrained_world),
          label_origin + cv::Point{0, 60}, {0, 255, 0});
      }
    }
  }

  std::ostringstream status;
  status << "frame=" << frame_index
         << " t=" << std::fixed << std::setprecision(3) << time_seconds
         << (mode == PlaybackMode::Realtime ? " realtime" : " offline")
         << " L2=" << std::setprecision(1) << l2_ms << "ms"
         << " L3=" << l3_ms << "ms"
         << " lag=" << realtime_lag_ms << "ms"
         << " skip=" << skipped_frames
         << " pred=" << cvRound(prediction_seconds * 1000.0) << "ms"
         << " geom=" << (geometry_constraints_enabled ? "on" : "off")
         << " det=" << detections.size()
         << " obs=" << observations.size()
         << " target=" << targets.size();
  if (selected_robot) {
    status << " selected=" << *selected_robot;
  }
  if (paused) {
    status << " [PAUSED]";
  }
  drawText(image, status.str(), {12, 28}, {255, 255, 255}, 0.6);

  int text_y = 55;
  for (const auto& target : targets) {
    const double second_radius = target.radius
      + (target.model == L3Estimation::TargetModel::FourArmorVehicle
           ? target.radius_offset
           : 0.0);
    std::ostringstream line;
    line << "robot=" << target.robot_id << ' '
         << trackerStateName(target.tracker_state)
         << " c=" << std::fixed << std::setprecision(2)
         << target.center.transpose()
         << " v=" << target.velocity.transpose()
         << " yaw=" << std::setprecision(1)
         << target.yaw * kRadToDeg << "deg"
         << " w=" << std::setprecision(2) << target.yaw_rate
         << " r=" << target.radius << '/' << second_radius;
    if (target.model == L3Estimation::TargetModel::FourArmorVehicle) {
      line << " angle=" << std::setprecision(1)
           << L3Estimation::fourArmorMinimumCornerAngle(
                target.radius, second_radius) * kRadToDeg
           << "deg";
    }
    drawText(
      image, line.str(), {12, text_y},
      selected_robot && target.robot_id == *selected_robot
        ? cv::Scalar{0, 255, 0}
        : cv::Scalar{220, 220, 220});
    text_y += 22;
  }

  drawText(
    image,
    "Vehicle: green=updated orange=predict-only magenta=future",
    {12, image.rows - 38}, {255, 255, 255});
  drawText(
    image,
    "Space: pause/resume  N/Right: step  L: armor text  Q/Esc: quit",
    {12, image.rows - 16}, {255, 255, 255});
}

const L3Estimation::ArmorObservation* selectObservation(
  const std::vector<L3Estimation::ArmorObservation>& observations,
  int robot_id)
{
  const L3Estimation::ArmorObservation* best = nullptr;
  for (const auto& observation : observations) {
    if (observation.robot_id == robot_id
        && (best == nullptr || observation.confidence > best->confidence)) {
      best = &observation;
    }
  }
  return best;
}

const L3Estimation::TargetState* selectTarget(
  const std::vector<L3Estimation::TargetState>& targets,
  int robot_id)
{
  const auto target = std::find_if(
    targets.begin(), targets.end(),
    [robot_id](const auto& value) { return value.robot_id == robot_id; });
  return target == targets.end() ? nullptr : &*target;
}

AdvanceAction handlePlayback(
  PlaybackMode mode,
  bool gui_enabled,
  bool& paused,
  bool& show_armor_text,
  double current_offset,
  double next_offset,
  Clock::time_point& playback_origin)
{
  if (!gui_enabled) {
    if (mode == PlaybackMode::Realtime) {
      std::this_thread::sleep_until(
        playback_origin
        + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>{next_offset}));
    }
    return AdvanceAction::Timed;
  }

  while (true) {
    const auto deadline = playback_origin
      + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>{next_offset});
    const auto now = Clock::now();
    const int delay_ms = paused
      ? 30
      : mode == PlaybackMode::Offline
      ? 1
      : std::clamp(
          static_cast<int>(std::ceil(
            std::chrono::duration<double, std::milli>(deadline - now).count())),
          1, 30);
    const int key = cv::waitKeyEx(delay_ms);
    if (key == 'q' || key == 'Q' || key == 27) {
      return AdvanceAction::Quit;
    }
    if (key == 'l' || key == 'L') {
      show_armor_text = !show_armor_text;
      return AdvanceAction::Redraw;
    }
    if (key == ' ') {
      paused = !paused;
      if (!paused) {
        playback_origin = Clock::now()
          - std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>{current_offset});
      }
      continue;
    }
    if (paused && (key == 'n' || key == 'N' || key == 65363)) {
      return AdvanceAction::Step;
    }
    if (!paused
        && (mode == PlaybackMode::Offline || Clock::now() >= deadline)) {
      return AdvanceAction::Timed;
    }
  }
}

int run(int argc, char** argv)
{
  cv::CommandLineParser cli(argc, argv, kCommandLineKeys);
  cli.about("Replay record_capture AVI/TXT through L2 and L3");
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return 2;
  }

  const auto base = recordBase(cli.get<std::string>(0));
  const auto video_path = std::filesystem::path{base.string() + ".avi"};
  const auto pose_path = std::filesystem::path{base.string() + ".txt"};
  const auto camera_config =
    std::filesystem::path{cli.get<std::string>("camera-config")};
  const auto l3_config =
    std::filesystem::path{cli.get<std::string>("l3-config")};
  const auto model_path =
    std::filesystem::path{cli.get<std::string>("model-path")};
  const std::string device = cli.get<std::string>("device");
  const std::string mode_text = cli.get<std::string>("mode");
  const PlaybackMode mode = mode_text == "realtime"
    ? PlaybackMode::Realtime
    : mode_text == "offline"
    ? PlaybackMode::Offline
    : throw std::invalid_argument("mode must be realtime or offline");
  const int start_index = cli.get<int>("start-index");
  const int prediction_ms = cli.get<int>("prediction-ms");
  const int requested_end = cli.get<int>("end-index");
  const int requested_robot = cli.get<int>("robot-id");
  const int plot_port = cli.get<int>("plotjuggler-port");
  const bool gui_enabled = !cli.has("no-gui");
  const bool plot_enabled = !cli.has("no-plotjuggler");
  bool show_armor_text = cli.has("show-armor-text");
  if (base.empty() || start_index < 0 || requested_end < 0
      || (requested_end > 0 && requested_end < start_index)
      || prediction_ms < 0 || prediction_ms > 2000
      || requested_robot < -1 || plot_port <= 0 || plot_port > 65535) {
    throw std::invalid_argument(
      "invalid input path, frame range, prediction horizon, robot ID, or UDP port");
  }

  const auto records = loadPoseRecords(pose_path);
  cv::VideoCapture video{video_path.string()};
  if (!video.isOpened()) {
    throw std::runtime_error("failed to open video: " + video_path.string());
  }
  const auto video_frames = static_cast<std::size_t>(
    std::llround(video.get(cv::CAP_PROP_FRAME_COUNT)));
  if (video_frames == 0 || video_frames != records.size()) {
    throw std::runtime_error(
      "video/pose count mismatch: video=" + std::to_string(video_frames)
      + " pose=" + std::to_string(records.size()));
  }
  if (static_cast<std::size_t>(start_index) >= records.size()) {
    throw std::invalid_argument("start-index is past the end of the recording");
  }
  const int end_index = requested_end == 0
    ? static_cast<int>(records.size()) - 1
    : requested_end;
  if (static_cast<std::size_t>(end_index) >= records.size()) {
    throw std::invalid_argument("end-index is past the end of the recording");
  }

  const auto calibration = loadCalibration(camera_config);
  const cv::Size video_size{
    cvRound(video.get(cv::CAP_PROP_FRAME_WIDTH)),
    cvRound(video.get(cv::CAP_PROP_FRAME_HEIGHT))};
  if (!calibration.matchesImageSize(video_size)) {
    throw std::runtime_error("video resolution does not match camera calibration");
  }
  if (start_index > 0 && !video.set(cv::CAP_PROP_POS_FRAMES, start_index)) {
    throw std::runtime_error("failed to seek video to start-index");
  }

  auto detector = makeDetector(model_path, device);
  auto l3_parameters = L3Estimation::loadL3Config(l3_config);
  if (cli.has("no-geometry-constraints")) {
    l3_parameters.tracker.enable_vehicle_geometry_constraints = false;
  }
  if (cli.has("no-ippe-dual-candidates")) {
    l3_parameters.pnp.enable_ippe_dual_candidates = false;
  }
  if (cli.has("no-predicted-face-yaw-selection")) {
    l3_parameters.pnp.enable_predicted_face_yaw_selection = false;
  }
  const bool geometry_constraints_enabled =
    l3_parameters.tracker.enable_vehicle_geometry_constraints;
  L3Estimation::TimePoint current_timestamp{};
  std::optional<Eigen::Quaterniond> current_pose;
  L3Estimation::TargetEstimator estimator{
    calibration,
    [&current_timestamp, &current_pose](L3Estimation::TimePoint timestamp) {
      return timestamp == current_timestamp
               ? current_pose
               : std::optional<Eigen::Quaterniond>{};
    },
    l3_parameters};

  std::unique_ptr<L6Telemetry::UdpJsonSender> sender;
  if (plot_enabled) {
    sender = std::make_unique<L6Telemetry::UdpJsonSender>(
      cli.get<std::string>("plotjuggler-host"),
      static_cast<std::uint16_t>(plot_port));
    std::cout << "PlotJuggler UDP JSON -> " << sender->host()
              << ':' << sender->port() << '\n';
  }
  if (gui_enabled) {
    cv::namedWindow("L3 video replay", cv::WINDOW_NORMAL);
    cv::resizeWindow("L3 video replay", 960, 720);
  }

  std::optional<int> selected_robot = requested_robot >= 0
    ? std::optional<int>{requested_robot}
    : std::nullopt;
  const double first_time = records[static_cast<std::size_t>(start_index)].time_seconds;
  const auto replay_epoch = Clock::now();
  auto playback_origin = Clock::now();
  bool paused = false;
  bool user_stopped = false;
  std::size_t processed = 0;
  std::size_t skipped_frames = 0;
  std::size_t published_target_samples = 0;
  double minimum_published_corner_angle =
    std::numeric_limits<double>::infinity();
  std::size_t udp_failures = 0;
  const auto run_begin = Clock::now();

  int frame_index = start_index;
  while (frame_index <= end_index) {
    cv::Mat frame;
    if (!video.read(frame) || frame.empty()) {
      throw std::runtime_error("video ended at frame " + std::to_string(frame_index));
    }
    if (frame.size() != calibration.image_size) {
      throw std::runtime_error("video resolution changed during replay");
    }

    const auto& record = records[static_cast<std::size_t>(frame_index)];
    current_pose = record.orientation;
    current_timestamp = replay_epoch
      + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>{record.time_seconds});

    const auto l2_begin = Clock::now();
    const auto detections = detector.detect(frame);
    const auto l2_end = Clock::now();
    const auto targets = estimator.update(
      detections,
      {.timestamp = current_timestamp, .image_size = frame.size()});
    const auto l3_end = Clock::now();
    const auto& observations = estimator.lastObservations();
    const double l2_ms =
      std::chrono::duration<double, std::milli>(l2_end - l2_begin).count();
    const double l3_ms =
      std::chrono::duration<double, std::milli>(l3_end - l2_end).count();
    const double current_offset = record.time_seconds - first_time;
    const auto scheduled_time = playback_origin
      + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>{current_offset});
    const double realtime_lag_ms = mode == PlaybackMode::Realtime
      ? std::max(
          0.0,
          std::chrono::duration<double, std::milli>(
            Clock::now() - scheduled_time).count())
      : 0.0;

    if (!selected_robot && !observations.empty()) {
      selected_robot = std::max_element(
        observations.begin(), observations.end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.confidence < rhs.confidence;
        })->robot_id;
      std::cout << "locked PlotJuggler robot ID " << *selected_robot << '\n';
    }
    const auto* observation = selected_robot
      ? selectObservation(observations, *selected_robot)
      : nullptr;
    const auto* target = selected_robot
      ? selectTarget(targets, *selected_robot)
      : nullptr;
    if (target != nullptr) {
      ++published_target_samples;
      if (target->model == L3Estimation::TargetModel::FourArmorVehicle) {
        minimum_published_corner_angle = std::min(
          minimum_published_corner_angle,
          L3Estimation::fourArmorMinimumCornerAngle(
            target->radius,
            target->radius + target->radius_offset));
      }
    }
    const Eigen::Vector3d gimbal_rpy =
      L6Telemetry::rotationToRpy(record.orientation.toRotationMatrix());

    if (sender && !sender->send(L6Telemetry::makeL3ReplayJson(
                    frame_index, record.time_seconds, gimbal_rpy,
                    observation, target, geometry_constraints_enabled,
                    l2_ms, l3_ms,
                    realtime_lag_ms, skipped_frames))) {
      ++udp_failures;
    }
    const auto showFrame = [&] {
      if (!gui_enabled) {
        return;
      }
      cv::Mat drawing = frame.clone();
      drawFrame(
        drawing, frame_index, record.time_seconds, l2_ms, l3_ms,
        detections, observations, targets, selected_robot,
        calibration, l3_parameters, record.orientation,
        static_cast<double>(prediction_ms) / 1000.0,
        geometry_constraints_enabled,
        mode, skipped_frames, realtime_lag_ms, show_armor_text, paused);
      cv::imshow("L3 video replay", drawing);
    };
    showFrame();
    ++processed;

    if (frame_index == end_index) {
      break;
    }
    const double next_offset =
      records[static_cast<std::size_t>(frame_index + 1)].time_seconds - first_time;
    AdvanceAction action;
    do {
      action = handlePlayback(
        mode, gui_enabled, paused, show_armor_text,
        current_offset, next_offset, playback_origin);
      if (action == AdvanceAction::Redraw) {
        showFrame();
      }
    } while (action == AdvanceAction::Redraw);
    if (action == AdvanceAction::Quit) {
      user_stopped = true;
      break;
    }

    int next_index = frame_index + 1;
    if (mode == PlaybackMode::Realtime && action == AdvanceAction::Timed) {
      const double elapsed =
        std::chrono::duration<double>(Clock::now() - playback_origin).count();
      while (next_index < end_index
             && records[static_cast<std::size_t>(next_index + 1)].time_seconds
                    - first_time <= elapsed) {
        ++next_index;
      }
    }
    while (frame_index + 1 < next_index) {
      if (!video.grab()) {
        throw std::runtime_error(
          "video ended while skipping frame " + std::to_string(frame_index + 1));
      }
      ++frame_index;
      ++skipped_frames;
    }
    ++frame_index;
  }

  if (gui_enabled) {
    cv::destroyWindow("L3 video replay");
  }
  const double wall_seconds =
    std::chrono::duration<double>(Clock::now() - run_begin).count();
  std::cout << (mode == PlaybackMode::Realtime ? "realtime" : "offline")
            << ", geometry constraints="
            << (geometry_constraints_enabled ? "on" : "off")
            << ": processed " << processed << " frames, skipped "
            << skipped_frames
            << (user_stopped ? " (stopped by user)" : "")
            << ", wall=" << std::fixed << std::setprecision(2)
            << wall_seconds << "s, processing FPS="
            << (wall_seconds > 0.0 ? processed / wall_seconds : 0.0)
            << ", target samples=" << published_target_samples;
  if (std::isfinite(minimum_published_corner_angle)) {
    std::cout << ", min corner angle="
              << minimum_published_corner_angle * kRadToDeg << "deg";
  }
  std::cout
            << ", UDP failures=" << udp_failures << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  L6Telemetry::initLogger();
  try {
    const int result = run(argc, argv);
    L6Telemetry::flushLogger();
    return result;
  } catch (const std::exception& error) {
    std::cerr << "L3 video replay failed: " << error.what() << '\n';
    L6Telemetry::flushLogger();
    return 1;
  }
}
