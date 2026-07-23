#include "l1_sensor/camera/camera_calibration.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/udp_json_sender.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

namespace {

using Clock = std::chrono::steady_clock;

const std::string kCommandLineKeys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{config-path c | ../tongjiceshi/configs/demo.yaml | demo.yaml 路径}"
  "{model-path m | model/armor_model/armor.xml | OpenVINO 装甲板模型路径}"
  "{device d | CPU | OpenVINO 推理设备}"
  "{start-index s | 0 | 视频起始帧下标}"
  "{end-index e | 0 | 视频结束帧下标，0 表示直到输入结束}"
  "{print-every p | 30 | 每隔多少帧打印状态，0 表示只打印总结}"
  "{show | | 显示检测框与 TargetState，按 q 或 ESC 退出}"
  "{plotjuggler | | 通过 UDP JSON 向 PlotJuggler 发送每帧 L3 状态}"
  "{plotjuggler-host | 127.0.0.1 | PlotJuggler UDP 目标 IPv4 地址}"
  "{plotjuggler-port | 9870 | PlotJuggler UDP 目标端口}"
  "{@input-path | ../tongjiceshi/assets/demo/demo | 不带扩展名的 AVI/TXT 路径}";

struct DemoRecord {
  double time_seconds = 0.0;
  Eigen::Quaterniond imu_orientation = Eigen::Quaterniond::Identity();
};

struct ReplayStatistics {
  std::size_t processed_frames = 0;
  std::size_t frames_with_detections = 0;
  std::size_t total_detections = 0;
  std::size_t max_detections = 0;
  std::size_t frames_with_targets = 0;
  std::size_t total_published_targets = 0;
  std::size_t max_targets = 0;
  std::size_t timestamp_gaps = 0;
  std::size_t invalid_target_states = 0;
  std::size_t plotjuggler_packets = 0;
  std::size_t plotjuggler_send_failures = 0;
  double total_l2_ms = 0.0;
  double total_l3_ms = 0.0;
  std::set<int> published_robot_ids;
};

std::vector<double> requiredVector(
  const YAML::Node& config,
  const std::string& key,
  std::size_t expected_size)
{
  const YAML::Node values = config[key];
  if (!values || !values.IsSequence() || values.size() != expected_size) {
    throw std::runtime_error(
      "demo config field '" + key + "' must contain "
      + std::to_string(expected_size) + " values");
  }

  std::vector<double> result;
  result.reserve(expected_size);
  for (const auto& value : values) {
    result.push_back(value.as<double>());
  }
  return result;
}

Eigen::Matrix3d matrix3FromConfig(
  const YAML::Node& config,
  const std::string& key)
{
  const std::vector<double> values = requiredVector(config, key, 9);
  return Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
    values.data());
}

Eigen::Vector3d vector3FromConfig(
  const YAML::Node& config,
  const std::string& key)
{
  const std::vector<double> values = requiredVector(config, key, 3);
  return Eigen::Map<const Eigen::Vector3d>(values.data());
}

L1Sensor::CameraCalibration calibrationFromConfig(
  const YAML::Node& config,
  const cv::Size& image_size)
{
  const std::vector<double> camera_values =
    requiredVector(config, "camera_matrix", 9);
  const std::vector<double> distortion_values =
    requiredVector(config, "distort_coeffs", 5);

  L1Sensor::CameraCalibration calibration;
  calibration.image_size = image_size;
  calibration.camera_matrix = cv::Mat(
    3, 3, CV_64FC1, const_cast<double*>(camera_values.data())).clone();
  calibration.distortion_coefficients = cv::Mat(
    1,
    static_cast<int>(distortion_values.size()),
    CV_64FC1,
    const_cast<double*>(distortion_values.data())).clone();
  return calibration;
}

bool readDemoRecord(std::istream& input, DemoRecord& record)
{
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (!(input >> record.time_seconds >> w >> x >> y >> z)) {
    return false;
  }

  record.imu_orientation = Eigen::Quaterniond{w, x, y, z};
  return true;
}

std::optional<Eigen::Quaterniond> gimbalToWorldRotation(
  const DemoRecord& record,
  const Eigen::Matrix3d& rotation_gimbal_to_imu_body)
{
  if (!std::isfinite(record.time_seconds)
      || !record.imu_orientation.coeffs().allFinite()
      || record.imu_orientation.norm() <= 1e-9) {
    return std::nullopt;
  }

  // 与 tongjiceshi::Solver::set_R_gimbal2world() 保持完全相同的语义。
  const Eigen::Matrix3d rotation_imu_body_to_world =
    record.imu_orientation.normalized().toRotationMatrix();
  const Eigen::Matrix3d rotation_gimbal_to_world =
    rotation_gimbal_to_imu_body.transpose()
    * rotation_imu_body_to_world
    * rotation_gimbal_to_imu_body;

  Eigen::Quaterniond result{rotation_gimbal_to_world};
  if (!result.coeffs().allFinite() || result.norm() <= 1e-9) {
    return std::nullopt;
  }
  return result.normalized();
}

L2Perception::ArmorDetector makeArmorDetector(
  const std::filesystem::path& model_path,
  const std::string& device)
{
  auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
  L2Perception::InferenceModelConfig model_config;
  model_config.model_path = model_path;
  model_config.device = device;
  // armor.xml 的宿主输入来自 OpenCV BGR 图像，送入模型前必须转换为 RGB。
  model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
  model_config.normalization_divisor = 255.0F;
  backend->load(model_config);

  L2Perception::ArmorDetector detector{std::move(backend)};
  if (!detector.ready()) {
    throw std::runtime_error("armor detector did not become ready");
  }
  return detector;
}

bool validTargetState(const L3Estimation::TargetState& target)
{
  return target.robot_id >= 0
         && target.center.allFinite()
         && target.velocity.allFinite()
         && std::isfinite(target.yaw)
         && std::isfinite(target.yaw_rate)
         && std::isfinite(target.radius)
         && std::isfinite(target.radius_offset)
         && std::isfinite(target.height_offset)
         && target.covariance.allFinite()
         && (target.covariance - target.covariance.transpose()).norm() < 1e-7
         && target.timestamp != L3Estimation::TimePoint{};
}

std::string armorColorName(L2Perception::ArmorColor color)
{
  switch (color) {
  case L2Perception::ArmorColor::Red:
    return "red";
  case L2Perception::ArmorColor::Blue:
    return "blue";
  case L2Perception::ArmorColor::Unknown:
    return "unknown";
  }
  return "unknown";
}

void drawReplay(
  cv::Mat& image,
  int frame_index,
  const std::vector<L2Perception::ArmorDetection>& detections,
  const std::vector<L3Estimation::TargetState>& targets)
{
  for (const auto& detection : detections) {
    const cv::Scalar color = detection.color == L2Perception::ArmorColor::Red
                               ? cv::Scalar{0, 0, 255}
                               : detection.color == L2Perception::ArmorColor::Blue
                                   ? cv::Scalar{255, 0, 0}
                                   : cv::Scalar{0, 255, 255};
    for (std::size_t index = 0; index < detection.corners.size(); ++index) {
      const cv::Point start{
        cvRound(detection.corners[index].x),
        cvRound(detection.corners[index].y)};
      const auto& next = detection.corners[(index + 1) % detection.corners.size()];
      const cv::Point end{cvRound(next.x), cvRound(next.y)};
      cv::line(image, start, end, color, 2, cv::LINE_AA);
    }

    std::ostringstream label;
    label << armorColorName(detection.color)
          << " id=" << detection.class_id
          << " conf=" << std::fixed << std::setprecision(2)
          << detection.confidence;
    cv::putText(
      image,
      label.str(),
      {cvRound(detection.corners.front().x),
       cvRound(detection.corners.front().y) - 6},
      cv::FONT_HERSHEY_SIMPLEX,
      0.5,
      color,
      1,
      cv::LINE_AA);
  }

  cv::putText(
    image,
    "frame=" + std::to_string(frame_index)
      + " detections=" + std::to_string(detections.size())
      + " targets=" + std::to_string(targets.size()),
    {12, 28},
    cv::FONT_HERSHEY_SIMPLEX,
    0.65,
    {0, 255, 0},
    2,
    cv::LINE_AA);

  int text_y = 55;
  for (const auto& target : targets) {
    std::ostringstream text;
    text << "robot=" << target.robot_id
         << " c=(" << std::fixed << std::setprecision(2)
         << target.center.x() << ',' << target.center.y() << ','
         << target.center.z() << ") v=("
         << target.velocity.x() << ',' << target.velocity.y() << ','
         << target.velocity.z() << ") yaw=" << target.yaw
         << " w=" << target.yaw_rate;
    cv::putText(
      image,
      text.str(),
      {12, text_y},
      cv::FONT_HERSHEY_SIMPLEX,
      0.5,
      {0, 255, 0},
      1,
      cv::LINE_AA);
    text_y += 24;
  }
}

void printFrameState(
  int frame_index,
  double input_time,
  double l2_ms,
  double l3_ms,
  const std::vector<L2Perception::ArmorDetection>& detections,
  const std::vector<L3Estimation::TargetState>& targets)
{
  std::cout << '[' << frame_index << "] t=" << std::fixed
            << std::setprecision(3) << input_time
            << " detections=" << detections.size()
            << " targets=" << targets.size()
            << " L2=" << std::setprecision(2) << l2_ms << "ms"
            << " L3=" << l3_ms << "ms\n";

  for (const auto& target : targets) {
    std::cout << "  robot=" << target.robot_id
              << " center=(" << target.center.transpose() << ')'
              << " velocity=(" << target.velocity.transpose() << ')'
              << " yaw=" << target.yaw
              << " yaw_rate=" << target.yaw_rate
              << " r1=" << target.radius
              << " dr=" << target.radius_offset
              << " dz=" << target.height_offset << '\n';
  }
}

nlohmann::json makePlotJugglerPayload(
  int frame_index,
  double input_time,
  double l2_ms,
  double l3_ms,
  const std::vector<L2Perception::ArmorDetection>& detections,
  const std::vector<L3Estimation::TargetState>& targets)
{
  nlohmann::json payload = {
    {"time", input_time},
    {"frame_index", frame_index},
    {"detection_count", detections.size()},
    {"target_count", targets.size()},
    {"processing_ms", {{"l2", l2_ms}, {"l3", l3_ms}}},
    {"targets", nlohmann::json::object()},
  };

  for (const auto& target : targets) {
    nlohmann::json covariance = nlohmann::json::object();
    for (Eigen::Index row = 0; row < target.covariance.rows(); ++row) {
      for (Eigen::Index column = 0; column < target.covariance.cols(); ++column) {
        covariance[
          "p_" + std::to_string(row) + '_' + std::to_string(column)] =
          target.covariance(row, column);
      }
    }

    payload["targets"]["robot_" + std::to_string(target.robot_id)] = {
      {"robot_id", target.robot_id},
      {"center",
       {{"x", target.center.x()},
        {"y", target.center.y()},
        {"z", target.center.z()}}},
      {"velocity",
       {{"x", target.velocity.x()},
        {"y", target.velocity.y()},
        {"z", target.velocity.z()}}},
      {"yaw", target.yaw},
      {"yaw_rate", target.yaw_rate},
      {"radius", target.radius},
      {"radius_offset", target.radius_offset},
      {"height_offset", target.height_offset},
      {"covariance", std::move(covariance)},
    };
  }

  return payload;
}

}  // namespace

int main(int argc, char** argv)
{
  L6Telemetry::initLogger();
  try {
    cv::CommandLineParser command_line(argc, argv, kCommandLineKeys);
    if (command_line.has("help")) {
      command_line.printMessage();
      return 0;
    }
    if (!command_line.check()) {
      command_line.printErrors();
      return 2;
    }

    const std::filesystem::path input_base =
      command_line.get<std::string>(0);
    const std::filesystem::path video_path = input_base.string() + ".avi";
    const std::filesystem::path pose_path = input_base.string() + ".txt";
    const std::filesystem::path config_path =
      command_line.get<std::string>("config-path");
    const std::filesystem::path model_path =
      command_line.get<std::string>("model-path");
    const std::string device = command_line.get<std::string>("device");
    const int start_index = command_line.get<int>("start-index");
    const int end_index = command_line.get<int>("end-index");
    const int print_every = command_line.get<int>("print-every");
    const bool show = command_line.has("show");
    const bool enable_plotjuggler = command_line.has("plotjuggler");
    const std::string plotjuggler_host =
      command_line.get<std::string>("plotjuggler-host");
    const int plotjuggler_port = command_line.get<int>("plotjuggler-port");

    if (start_index < 0 || end_index < 0
        || (end_index > 0 && end_index < start_index)
        || print_every < 0
        || plotjuggler_port <= 0
        || plotjuggler_port > 65535) {
      throw std::invalid_argument("invalid start/end/print interval");
    }

    cv::VideoCapture video{video_path.string()};
    if (!video.isOpened()) {
      throw std::runtime_error("failed to open demo video: " + video_path.string());
    }
    const double video_fps = video.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(video_fps) || video_fps <= 0.0) {
      throw std::runtime_error("demo video does not contain a valid frame rate");
    }
    std::ifstream pose_stream{pose_path};
    if (!pose_stream.is_open()) {
      throw std::runtime_error("failed to open demo pose text: " + pose_path.string());
    }

    if (!video.set(cv::CAP_PROP_POS_FRAMES, start_index)) {
      throw std::runtime_error("failed to seek demo video");
    }
    DemoRecord skipped_record;
    for (int index = 0; index < start_index; ++index) {
      if (!readDemoRecord(pose_stream, skipped_record)) {
        throw std::runtime_error("pose text ended before start-index");
      }
    }

    cv::Mat frame;
    DemoRecord record;
    if (!video.read(frame) || frame.empty()) {
      throw std::runtime_error("demo video has no frame at start-index");
    }
    if (!readDemoRecord(pose_stream, record)) {
      throw std::runtime_error("demo pose text has no record at start-index");
    }

    const YAML::Node config = YAML::LoadFile(config_path.string());
    const L1Sensor::CameraCalibration calibration =
      calibrationFromConfig(config, frame.size());
    const Eigen::Matrix3d rotation_camera_to_gimbal =
      matrix3FromConfig(config, "R_camera2gimbal");
    const Eigen::Vector3d translation_camera_to_gimbal =
      vector3FromConfig(config, "t_camera2gimbal");
    const Eigen::Matrix3d rotation_gimbal_to_imu_body =
      matrix3FromConfig(config, "R_gimbal2imubody");

    L2Perception::ArmorDetector armor_detector =
      makeArmorDetector(model_path, device);

    L3Estimation::TimePoint current_timestamp{};
    std::optional<Eigen::Quaterniond> current_gimbal_pose;
    const L3Estimation::GimbalPoseProvider pose_provider =
      [&current_timestamp, &current_gimbal_pose](L3Estimation::TimePoint timestamp)
      -> std::optional<Eigen::Quaterniond> {
      if (timestamp != current_timestamp) {
        return std::nullopt;
      }
      return current_gimbal_pose;
    };

    L3Estimation::TargetEstimator target_estimator{
      calibration,
      rotation_camera_to_gimbal,
      translation_camera_to_gimbal,
      pose_provider};

    std::unique_ptr<L6Telemetry::UdpJsonSender> plotjuggler_sender;
    if (enable_plotjuggler) {
      plotjuggler_sender = std::make_unique<L6Telemetry::UdpJsonSender>(
        plotjuggler_host,
        static_cast<std::uint16_t>(plotjuggler_port));
      std::cout << "PlotJuggler UDP JSON: "
                << plotjuggler_sender->host() << ':'
                << plotjuggler_sender->port() << '\n';
    }

    if (show) {
      cv::namedWindow("target estimator demo replay", cv::WINDOW_NORMAL);
    }

    ReplayStatistics statistics;
    const Clock::time_point replay_epoch = Clock::now();
    std::optional<double> previous_input_time;
    std::string stop_reason = "requested end-index";
    bool user_stopped = false;
    const Clock::time_point playback_begin = Clock::now();

    for (int frame_index = start_index;; ++frame_index) {
      if (end_index > 0 && frame_index > end_index) {
        break;
      }
      if (frame.size() != calibration.image_size) {
        throw std::runtime_error("demo video resolution changed during replay");
      }

      if (previous_input_time) {
        const double input_dt = record.time_seconds - *previous_input_time;
        if (!std::isfinite(input_dt) || input_dt <= 0.0) {
          throw std::runtime_error("demo timestamps are not strictly increasing");
        }
        if (input_dt > 0.1) {
          ++statistics.timestamp_gaps;
          std::cout << "[gap] frame=" << frame_index
                    << " dt=" << std::fixed << std::setprecision(3)
                    << input_dt << "s; trackers will reset\n";
        }
      }
      previous_input_time = record.time_seconds;

      current_timestamp = replay_epoch
        + std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>{record.time_seconds});
      current_gimbal_pose = gimbalToWorldRotation(
        record, rotation_gimbal_to_imu_body);

      const auto l2_begin = Clock::now();
      const auto detections = armor_detector.detect(frame);
      const auto l2_end = Clock::now();
      const auto targets = target_estimator.update(detections, current_timestamp);
      const auto l3_end = Clock::now();

      const double l2_ms =
        std::chrono::duration<double, std::milli>(l2_end - l2_begin).count();
      const double l3_ms =
        std::chrono::duration<double, std::milli>(l3_end - l2_end).count();
      ++statistics.processed_frames;
      statistics.total_l2_ms += l2_ms;
      statistics.total_l3_ms += l3_ms;
      statistics.total_detections += detections.size();
      statistics.max_detections =
        std::max(statistics.max_detections, detections.size());
      statistics.total_published_targets += targets.size();
      statistics.max_targets = std::max(statistics.max_targets, targets.size());
      if (!detections.empty()) {
        ++statistics.frames_with_detections;
      }
      if (!targets.empty()) {
        ++statistics.frames_with_targets;
      }
      for (const auto& target : targets) {
        statistics.published_robot_ids.insert(target.robot_id);
        if (!validTargetState(target)) {
          ++statistics.invalid_target_states;
        }
      }

      if (plotjuggler_sender) {
        const nlohmann::json payload = makePlotJugglerPayload(
          frame_index,
          record.time_seconds,
          l2_ms,
          l3_ms,
          detections,
          targets);
        if (plotjuggler_sender->send(payload)) {
          ++statistics.plotjuggler_packets;
        } else {
          ++statistics.plotjuggler_send_failures;
        }
      }

      if (print_every > 0
          && ((frame_index - start_index) % print_every == 0)) {
        printFrameState(
          frame_index,
          record.time_seconds,
          l2_ms,
          l3_ms,
          detections,
          targets);
      }

      if (show) {
        cv::Mat drawing = frame.clone();
        drawReplay(drawing, frame_index, detections, targets);
        cv::resize(drawing, drawing, {}, 0.5, 0.5, cv::INTER_AREA);
        cv::imshow("target estimator demo replay", drawing);

        // 以视频帧率为准设置绝对播放时刻，并扣除本帧检测、估计和绘图耗时。
        // 使用绝对时刻还能避免每帧毫秒取整造成的累计播放漂移。
        const auto playback_deadline = playback_begin
          + std::chrono::duration<double>{
            static_cast<double>(statistics.processed_frames) / video_fps};
        const auto remaining = playback_deadline - Clock::now();
        const int wait_ms = remaining > Clock::duration::zero()
          ? std::max(
              1,
              static_cast<int>(
                std::chrono::ceil<std::chrono::milliseconds>(remaining).count()))
          : 1;
        const int key = cv::waitKey(wait_ms);
        if (key == 'q' || key == 'Q' || key == 27) {
          stop_reason = "user requested stop";
          user_stopped = true;
          break;
        }
      }

      if (!video.read(frame) || frame.empty()) {
        stop_reason = "video exhausted";
        break;
      }
      if (!readDemoRecord(pose_stream, record)) {
        stop_reason = "pose text exhausted";
        break;
      }
    }

    if (show) {
      cv::destroyWindow("target estimator demo replay");
    }

    const double frame_count = static_cast<double>(statistics.processed_frames);
    std::cout << "\nTargetEstimator demo replay summary\n"
              << "stop reason: " << stop_reason << '\n'
              << "processed frames: " << statistics.processed_frames << '\n'
              << "frames with detections: " << statistics.frames_with_detections << '\n'
              << "total detections: " << statistics.total_detections << '\n'
              << "max detections per frame: " << statistics.max_detections << '\n'
              << "frames with published targets: " << statistics.frames_with_targets << '\n'
              << "total published target samples: "
              << statistics.total_published_targets << '\n'
              << "max published targets per frame: " << statistics.max_targets << '\n'
              << "timestamp gaps over 100ms: " << statistics.timestamp_gaps << '\n'
              << "invalid target states: " << statistics.invalid_target_states << '\n'
              << "PlotJuggler packets sent: "
              << statistics.plotjuggler_packets << '\n'
              << "PlotJuggler send failures: "
              << statistics.plotjuggler_send_failures << '\n'
              << "video frame rate: " << video_fps << " FPS\n"
              << "average L2 time: "
              << (frame_count > 0.0 ? statistics.total_l2_ms / frame_count : 0.0)
              << " ms\n"
              << "average L3 time: "
              << (frame_count > 0.0 ? statistics.total_l3_ms / frame_count : 0.0)
              << " ms\n"
              << "published robot ids:";
    for (const int robot_id : statistics.published_robot_ids) {
      std::cout << ' ' << robot_id;
    }
    std::cout << '\n';

    L6Telemetry::flushLogger();
    if (statistics.invalid_target_states != 0) {
      return 3;
    }
    if (!user_stopped && statistics.processed_frames == 0) {
      return 4;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "TargetEstimator demo replay failed: " << error.what() << '\n';
    L6Telemetry::flushLogger();
    return 1;
  }
}
