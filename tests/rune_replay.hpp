#pragma once

// 大小符回放共用实现，模式由各自的可执行入口固定。
#include "runtime/auto_aim_config.hpp"
#include "rune/debug_overlay.hpp"
#include "armor/l5_telemetry/aim_overlay.hpp"
#include "armor/l5_telemetry/fps_counter.hpp"
#include "armor/l5_telemetry/udp_json_sender.hpp"
#include "sensor/camera/camera_calibration.hpp"
#include "sensor/serial/serial_config.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {
const std::string kKeys =
  "{help h usage ? | false | 输出命令行参数说明}"
  "{config | config/auto_aim.yaml | 算法配置，读取 rune 节点}"
  "{planner-config | config/planner_config.yaml | 弹道、延迟和 MPC 配置}"
  "{calibration c | config/camera_config.yaml | 相机标定 yaml}"
  "{enemy | blue | 目标颜色：red / blue}"
  "{model m | auto | auto 使用 rune.detector.model_path，否则覆盖模型路径}"
  "{device d | auto | auto 使用 rune.detector.device，否则覆盖推理设备}"
  "{conf | 0 | 覆盖检测置信度，0 保持配置}"
  "{pose | auto | auto 读取同名 txt，无配套姿态时做相机相对视觉预测 / none 仅检测 / 指定姿态 txt}"
  "{preview-ms | 200 | 无有效靶板拦截规划时的视觉预测提前量（毫秒）}"
  "{convention | imu | 录像四元数约定：imu / sp，与装甲板回放一致}"
  "{serial-config | config/serial_config.yaml | imu 约定的 R_imu_barrel}"
  "{bullet-speed | 27.0 | 回放弹速（m/s）}"
  "{start-index s | 0 | 起始帧下标}"
  "{end-index e | 0 | 结束帧下标（含），0 表示到结尾}"
  "{wait w | 30 | 每帧等待毫秒；0 手动逐帧，空格暂停，q/ESC 退出}"
  "{output | | 保存最后一帧标注图的路径}";

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

const char* statusName(Rune::Status status) {
  switch (status) {
  case Rune::Status::Ready: return "ready";
  case Rune::Status::NoDetection: return "no_detection";
  case Rune::Status::InvalidInput: return "invalid_input";
  case Rune::Status::NoContour: return "no_contour";
  case Rune::Status::PoseFailed: return "pose_failed";
  case Rune::Status::MotionNotReady: return "motion_not_ready";
  case Rune::Status::BallisticFailed: return "ballistic_failed";
  case Rune::Status::AimNotReady: return "aim_not_ready";
  case Rune::Status::AlreadyFired: return "already_fired";
  case Rune::Status::WarmingUp: return "warming_up";
  case Rune::Status::CenterHold: return "center_hold";
  case Rune::Status::Idle: return "idle";
  case Rune::Status::ShotCooldown: return "shot_cooldown";
  case Rune::Status::ShootDisabled: return "shoot_disabled";
  case Rune::Status::InvalidFeedback: return "invalid_feedback";
  case Rune::Status::CommandJump: return "command_jump";
  case Rune::Status::PlanStale: return "plan_stale";
  }
  return "unknown";
}

struct PoseSample {
  double seconds{};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

bool readPose(std::istream& text, PoseSample& pose) {
  text >> std::ws;
  if (text.eof()) return false;
  double w{}, x{}, y{}, z{};
  require(static_cast<bool>(text >> pose.seconds >> w >> x >> y >> z),
          "姿态文本必须为每行 t w x y z");
  pose.orientation = Eigen::Quaterniond{w, x, y, z};
  require(std::isfinite(pose.seconds) && pose.seconds >= 0 &&
          pose.orientation.coeffs().allFinite() && pose.orientation.squaredNorm() > 1e-12,
          "姿态文本含无效时间或四元数");
  return true;
}

Eigen::Quaterniond worldBarrelPose(const PoseSample& pose, bool sp,
                                  const Eigen::Matrix3d& R_imu_barrel) {
  const Eigen::Matrix3d rotation = pose.orientation.normalized().toRotationMatrix();
  if (sp) {
    Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
    flip(0, 0) = flip(1, 1) = -1;
    return Eigen::Quaterniond{flip.transpose() * rotation * flip};
  }
  return Eigen::Quaterniond{rotation * R_imu_barrel};
}
}  // namespace

inline int runRuneReplay(int argc, char** argv, Rune::Mode mode,
                         const char* program_name, const char* default_input) {
  try {
    const auto keys = kKeys + "{@input-path | " + default_input +
        " | 视频路径或不含扩展名的路径}";
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.get<bool>("help")) { cli.printMessage(); return 0; }
    const auto config_path = cli.get<std::string>("config");
    const auto planner_path = cli.get<std::string>("planner-config");
    const auto calibration_path = cli.get<std::string>("calibration");
    const auto enemy = cli.get<std::string>("enemy");
    const auto model = cli.get<std::string>("model");
    const auto device = cli.get<std::string>("device");
    const float confidence = cli.get<float>("conf");
    const auto pose_option = cli.get<std::string>("pose");
    const double preview_s = cli.get<double>("preview-ms") / 1000;
    const auto convention = cli.get<std::string>("convention");
    const auto serial_path = cli.get<std::string>("serial-config");
    const double bullet_speed = cli.get<double>("bullet-speed");
    const int start_index = cli.get<int>("start-index");
    const int end_index = cli.get<int>("end-index");
    const int wait_ms = cli.get<int>("wait");
    const auto output = cli.get<std::string>("output");
    const std::filesystem::path input{cli.get<std::string>(0)};
    if (!cli.check()) { cli.printErrors(); return 1; }
    require(enemy == "red" || enemy == "blue", "enemy 必须是 red 或 blue");
    require(convention == "imu" || convention == "sp", "convention 必须是 imu 或 sp");
    require(start_index >= 0 && (end_index == 0 || end_index >= start_index), "起止帧下标无效");
    require(wait_ms >= 0, "wait 不能为负");
    require(std::isfinite(preview_s) && preview_s > 0 && preview_s <= 2,
            "preview-ms 必须在 (0, 2000] 内");
    require(std::isfinite(bullet_speed) && bullet_speed > 0, "bullet-speed 必须为正的有限值");
    require(std::isfinite(confidence) && confidence >= 0 && confidence <= 1, "conf 必须在 [0, 1] 内");

    const auto runtime_config = runtime::loadAutoAimConfig(config_path);
    const auto& debug = runtime_config.debug;
    auto config = runtime_config.rune;
    if (model != "auto") config.detector.model_path = model;
    if (device != "auto") config.detector.device = device;
    if (confidence > 0) config.detector.confidence_threshold = confidence;
    config.planning.shoot_enable = false;
    config.fire.shoot_enable = false;
    runtime::RuneRuntime rune(config, L4Planning::loadPlannerTuning(planner_path));
    // Separate observation-only instance: missing IMU data must not be replaced
    // with fabricated feedback or used to generate aiming/fire commands.
    Rune::Processor visual_rune(config.geometry);
    std::optional<L6Telemetry::UdpJsonSender> plotter;
    if (debug.plot)
      plotter.emplace(debug.plot_host, static_cast<std::uint16_t>(debug.plot_port));

    const auto yaml = YAML::LoadFile(calibration_path);
    const auto calibration = L1Sensor::loadCameraCalibration(yaml["calibration"], calibration_path);
    const auto video_path = input.has_extension() ? input.string() : input.string() + ".avi";
    auto pose_path = input;
    pose_path.replace_extension(".txt");
    if (pose_option != "auto" && pose_option != "none") pose_path = pose_option;
    std::ifstream poses;
    if (pose_option != "none") {
      poses.open(pose_path);
      require(pose_option == "auto" || poses.is_open(), "无法打开姿态文本：" + pose_path.string());
      require(!std::filesystem::exists(pose_path) || poses.is_open(),
              "无法读取姿态文本：" + pose_path.string());
    }
    const bool has_pose = poses.is_open();
    Eigen::Matrix3d R_imu_barrel = Eigen::Matrix3d::Identity();
    if (has_pose && convention == "imu")
      R_imu_barrel = L1Sensor::loadSerialConfig(serial_path).R_imu_barrel;
    cv::VideoCapture video(video_path);
    require(video.isOpened(), "无法打开录像：" + video_path);
    const double fps = video.get(cv::CAP_PROP_FPS);
    require(has_pose || (std::isfinite(fps) && fps > 0), "无姿态文本且视频 FPS 无效");
    if (start_index > 0)
      require(video.set(cv::CAP_PROP_POS_FRAMES, start_index), "无法跳转到 start-index");
    PoseSample pose;
    for (int i = 0; has_pose && i < start_index; ++i)
      require(readPose(poses, pose), "姿态文本在 start-index 之前结束");

    std::cout << "能量机关回放: " << video_path << " mode="
              << (mode == Rune::Mode::Small ? "small" : "big") << " enemy=" << enemy << '\n'
              << (has_pose ? "姿态文本: " + pose_path.string()
                           : pose_option == "none" ? "仅检测回放（--pose=none）"
                           : "无姿态文本，运行相机相对视觉预测；相机运动未补偿") << '\n'
              << "shoot_enable=false；不连接相机和串口\n";
    std::size_t frames = 0, positive_frames = 0, total_detections = 0, found_frames = 0, feasible_frames = 0;
    std::size_t overlay_frames = 0;
    std::size_t motion_ready_frames=0, history_resets=0, target_switches=0, prediction_frames=0;
    std::map<Rune::Status, std::size_t> statuses;
    const auto epoch = std::chrono::steady_clock::now();
    L6Telemetry::FpsCounter frame_rate(std::chrono::milliseconds(500));
    double previous_seconds = -1;
    double detect_ms = 0;
    bool paused = false;
    cv::Mat image, annotated;
    for (int index = start_index; end_index == 0 || index <= end_index; ++index) {
      if (!video.read(image) || image.empty()) break;
      if (has_pose && !readPose(poses, pose)) {
        std::cout << "姿态文本结束，停止回放配对帧\n";
        break;
      }
      const double seconds = has_pose ? pose.seconds : index / fps;
      require(seconds > previous_seconds, "姿态时间戳必须严格递增");
      previous_seconds = seconds;
      Rune::Frame frame;
      frame.bgr = image;
      frame.mode = mode;
      frame.color = enemy == "blue" ? Rune::Color::Blue : Rune::Color::Red;
      frame.captured_at = epoch + std::chrono::duration_cast<Rune::TimePoint::duration>(
          std::chrono::duration<double>(seconds));
      const bool calibrated = Rune::applyCameraCalibration(frame, calibration);
      if (has_pose) frame.q_world_barrel = worldBarrelPose(pose, convention == "sp", R_imu_barrel);
      const auto started = std::chrono::steady_clock::now();
      rune.detect(frame);
      detect_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
      ++frames;
      positive_frames += !frame.detections.empty();
      total_detections += frame.detections.size();
      const bool full_pipeline = has_pose && calibrated;
      const bool visual_only = !has_pose && calibrated && pose_option != "none";
      Rune::Status visual_status = Rune::Status::NoDetection;
      if (full_pipeline) {
        L1Sensor::RobotState state;
        state.mode = mode == Rune::Mode::Small ? L1Sensor::WorkMode::SmallBuff : L1Sensor::WorkMode::BigBuff;
        state.enemy_color = enemy == "blue" ? L1Sensor::EnemyColor::Blue : L1Sensor::EnemyColor::Red;
        state.bullet_speed = bullet_speed;
        state.timestamp = frame.captured_at;
        // 与装甲板回放一致：规划和反馈使用录像时刻，播放速度不影响模型预热。
        (void)rune.process(frame, state, frame.q_world_barrel, frame.captured_at);
        const auto& command = rune.lastCommand();
        found_frames += command.found;
        feasible_frames += command.fire_feasible;
        ++statuses[command.status];
      }
      if (visual_only) {
        visual_status = visual_rune.observe(frame);
        ++statuses[visual_status];
      }
      const std::optional<Rune::Motion> no_motion;
      const auto& motion = full_pipeline ? rune.lastMotion()
          : visual_only ? visual_rune.lastMotion() : no_motion;
      if (motion) {
        motion_ready_frames += motion->ready;
        history_resets += motion->pose.history_reset;
        target_switches += motion->pose.target_switched;
      }
      const std::string reason = !calibrated
          ? "DETECTION ONLY: calibration missing or image size mismatch"
          : visual_only ? std::string("VISUAL ONLY (camera motion uncompensated): ") + statusName(visual_status)
          : !has_pose ? "DETECTION ONLY: --pose=none"
          : std::string("status=") + statusName(rune.lastCommand().status);
      if (frames == 1 || frames % 100 == 0) {
        std::cout << "frame=" << index << " t=" << seconds << " det=" << frame.detections.size()
                  << ' ' << reason << '\n';
        if (const auto diagnostics = rune.detectorDiagnostics())
          std::cout << "  " << Rune::formatDetectionDiagnostics(*diagnostics) << '\n';
        if (motion)
          std::cout << "  motion ready=" << motion->ready
                    << " speed_rad_s=" << motion->angular_speed_rad_s
                    << " fit_rms_rad=" << std::sqrt(motion->fit_cost_rad2)
                    << " model=" << static_cast<int>(motion->model)
                    << " phase_rad=" << motion->pose.phase_rad << '\n';
      }
      if (plotter) {
        const auto& command = rune.lastCommand();
        nlohmann::json telemetry{
            {"t", seconds}, {"frame", index},
            {"mode", mode == Rune::Mode::Small ? 2 : 3},
            {"rune", {{"detections", frame.detections.size()},
                      {"has_pose", has_pose}, {"calibrated", calibrated},
                      {"full_pipeline", full_pipeline}, {"status", reason},
                      {"visual_only", visual_only}, {"motion_ready", motion && motion->ready},
                      {"found", full_pipeline && command.found},
                      {"fire_feasible", full_pipeline && command.fire_feasible},
                      {"shoot", false}, {"using_mpc", command.using_mpc},
                      {"yaw_rad", command.yaw_rad}, {"pitch_rad", command.pitch_rad}}}};
        if (const auto diagnostics = rune.detectorDiagnostics())
          telemetry["rune"]["detector_diagnostics"] = Rune::formatDetectionDiagnostics(*diagnostics);
        (void)plotter->send(telemetry);
      }
      const double playback_fps = frame_rate.update();
      const bool draw_overlay = debug.overlay && index % debug.overlay_every == 0;
      if (draw_overlay || !output.empty()) {
        annotated = image.clone();
        Rune::drawDetectionOverlay(annotated, frame.detections, mode);
        if (motion) {
          Rune::drawTrackingOverlay(annotated, frame, motion->pose);
          prediction_frames += Rune::drawPredictionOverlay(annotated, frame, *motion,
              config.geometry, full_pipeline && rune.lastPlan() ? &*rune.lastPlan() : nullptr,
              preview_s);
        }
        if (const auto diagnostics = rune.detectorDiagnostics())
          Rune::drawDetectionDiagnostics(annotated, *diagnostics);
        const std::string label = std::string(mode == Rune::Mode::Small ? "Small rune " : "Big rune ") + reason;
        cv::putText(annotated, label, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 255, 255}, 2);
        if (full_pipeline && rune.lastCommand().found) {
          const auto& command = rune.lastCommand();
          const double degrees = 180 / std::numbers::pi;
          cv::putText(annotated, "yaw=" + std::to_string(command.yaw_rad * degrees) +
                      " pitch=" + std::to_string(command.pitch_rad * degrees) +
                      (command.using_mpc ? " MPC" : ""),
                      {20, 100}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {0, 255, 255}, 2);
        }
        L6Telemetry::drawFpsOverlay(annotated, playback_fps);
      }
      if (!debug.overlay) continue;
      if (draw_overlay) {
        cv::Mat display;
        cv::resize(annotated, display, {}, 0.5, 0.5);
        cv::imshow(program_name, display);
        ++overlay_frames;
      }
      int key = cv::waitKey(wait_ms);
      if (key == 'q' || key == 27) break;
      if (key == ' ') {
        paused = true;
        while (paused) {
          key = cv::waitKey(0);
          if (key == 'q' || key == 27 || key == ' ') paused = false;
        }
        if (key == 'q' || key == 27) break;
      }
    }
    require(frames > 0, "指定范围内没有可回放的帧");
    if (!output.empty()) {
      const auto parent = std::filesystem::path(output).parent_path();
      if (!parent.empty()) std::filesystem::create_directories(parent);
      require(cv::imwrite(output, annotated), "无法保存标注图：" + output);
    }
    if (debug.overlay) cv::destroyAllWindows();
    std::cout << "回放结束: frames=" << frames << " positive_frames=" << positive_frames
              << " total_detections=" << total_detections << " found_frames=" << found_frames
              << " fire_feasible_frames=" << feasible_frames << " overlay_frames=" << overlay_frames
              << " motion_ready_frames=" << motion_ready_frames << " history_resets=" << history_resets
              << " target_switches=" << target_switches
              << " prediction_frames=" << prediction_frames
              << " mean_detect_ms=" << detect_ms / frames << '\n';
    for (const auto& [status, count] : statuses)
      std::cout << "  " << statusName(status) << ": " << count << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << program_name << " 失败: " << error.what() << '\n';
    return 1;
  }
}
