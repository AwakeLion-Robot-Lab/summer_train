#include "runtime/auto_aim_runtime.hpp"
#include "runtime/pose_detection_schedule.hpp"
#include "runtime/pose_roi.hpp"
#include "runtime/single_frame_pipeline.hpp"
#include "sensor/camera/camera.hpp"
#include "sensor/serial/serial_config.hpp"
#include "sensor/serial/serial_worker.hpp"
#include "l1_perception/armor.hpp"
#include "l1_perception/armor/armor_detector.hpp"
#include "l2_estimation/armor/eskf_tracker.hpp"
#include "l3_planning/planner.hpp"
#include "l3_planning/planner_config.hpp"
#include "l4_control/controller.hpp"
#include "l5_telemetry/aim_overlay.hpp"
#include "l5_telemetry/logger.hpp"
#include "l5_telemetry/math.hpp"
#include "l5_telemetry/udp_json_sender.hpp"
#include "runtime/auto_aim_config.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/l4_target_adapter.hpp"
#include "rune/debug_overlay.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point end, Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct FrameTiming {
  Clock::time_point captured_at{}, ready_at{}, queued_at{};
  double pose_wait_ms{0}, detect_ms{0}, estimate_ms{0}, plan_control_ms{0};
  unsigned detection_passes{0};
  bool pose_overlap{false}, pose_roi_repair{false}, pose_timeout{false};
  double result_wait_ms{0};
  std::uint64_t stale_results{0};
  std::optional<double> filter_input_age_ms, filter_output_age_ms;
  L2Perception::DetectTiming detection_stages{};
};

// Every field belongs to this frame. No references to live tracker/detector state
// or OpenVINO output tensors cross the handoff.
struct ArmorVisionPacket {
  cv::Mat image;
  FrameTiming timing;
  Clock::time_point completed_at{};
  std::optional<L1Sensor::RobotState> state;
  std::optional<Eigen::Quaterniond> image_pose;
  L2Perception::ArmorFrame perception;
  std::optional<L3Estimation::EskfTarget> target;
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  int match_count{0};
  std::uint64_t tracker_drops{0};
  bool has_tracker{false};
};

nlohmann::json pipelineTiming(const FrameTiming& timing,
                              const L1Sensor::SerialWorker& serial) {
  nlohmann::json data{
      {"frame_age_at_read_ms", milliseconds(timing.ready_at, timing.captured_at)},
      {"pose_wait_ms", timing.pose_wait_ms}, {"detect_ms", timing.detect_ms},
      {"detection_passes", timing.detection_passes},
      {"estimate_ms", timing.estimate_ms}, {"plan_control_ms", timing.plan_control_ms}};
  data["pose_overlap"] = timing.pose_overlap;
  data["pose_roi_repair"] = timing.pose_roi_repair;
  data["pose_timeout"] = timing.pose_timeout;
  data["result_wait_ms"] = timing.result_wait_ms;
  data["stale_results"] = timing.stale_results;
  if (timing.filter_input_age_ms) data["filter_input_age_ms"] = *timing.filter_input_age_ms;
  if (timing.filter_output_age_ms) data["filter_output_age_ms"] = *timing.filter_output_age_ms;
  data["preprocess_ms"] = timing.detection_stages.preprocess;
  data["infer_ms"] = timing.detection_stages.infer;
  data["decode_ms"] = timing.detection_stages.decode;
  data["refine_ms"] = timing.detection_stages.refine;
  data["side_light_ms"] = timing.detection_stages.side_light;
  if (timing.queued_at != Clock::time_point{}) {
    data["image_to_queue_ms"] = milliseconds(timing.queued_at, timing.captured_at);
    data["read_to_queue_ms"] = milliseconds(timing.queued_at, timing.ready_at);
  }
  // 串口线程异步发送；这组值属于最近一次已完成的命令，不与本帧耗时相加。
  if (const auto sent = serial.latestCommandTiming()) {
    data["serial_last_queue_ms"] = milliseconds(sent->write_started_at, sent->queued_at);
    data["serial_last_write_ms"] = milliseconds(sent->write_completed_at, sent->write_started_at);
    data["serial_last_sample_age_ms"] = milliseconds(Clock::now(), sent->write_completed_at);
  }
  return data;
}

// L1 的 enemy_color 是下位机给出的目标阵营；L2 的 ArmorColor 是图像识别结果。
// 任何一侧未知时都不允许作为自瞄目标，避免误击友军。
bool isEnemyArmor(L2Perception::ArmorColor observed,
                  L1Sensor::EnemyColor expected) noexcept {
  switch (expected) {
  case L1Sensor::EnemyColor::Red:
    return observed == L2Perception::ArmorColor::Red;
  case L1Sensor::EnemyColor::Blue:
    return observed == L2Perception::ArmorColor::Blue;
  case L1Sensor::EnemyColor::Unknown:
    return false;
  }

  return false;
}

L2Perception::ArmorColor enemyArmorColor(
  L1Sensor::EnemyColor expected) noexcept
{
  switch (expected) {
  case L1Sensor::EnemyColor::Red:
    return L2Perception::ArmorColor::Red;
  case L1Sensor::EnemyColor::Blue:
    return L2Perception::ArmorColor::Blue;
  case L1Sensor::EnemyColor::Unknown:
    return L2Perception::ArmorColor::Unknown;
  }
  return L2Perception::ArmorColor::Unknown;
}

// Shared by synchronous and pipelined execution. PoseSource is the serial worker
// in production; deterministic replay supplies the same timestamped pose interface.
template<class PoseSource, class Detector>
void processArmorVision(ArmorVisionPacket& packet,
                        Detector& armor_detector,
                        std::optional<L3Estimation::EskfTracker>& tracker,
                        const PoseSource& serial, bool serial_started, int pose_wait_ms,
                        runtime::PoseOverlapState* overlap = nullptr) {
  auto& frame = packet.image;
  auto& frame_timing = packet.timing;
  const auto timestamp = frame_timing.captured_at;
  const auto pose_deadline = frame_timing.ready_at +
      std::chrono::milliseconds(pose_wait_ms);
  const auto& state = packet.state;
  packet.has_tracker = tracker.has_value();
  if (!serial_started || !state) {
    if (tracker) tracker->reset();
    const auto started = Clock::now();
    packet.perception = armor_detector.detectFrame(frame);
    frame_timing.detect_ms = milliseconds(Clock::now(), started);
    frame_timing.detection_stages = armor_detector.lastTiming();
    frame_timing.detection_passes = 1;
    packet.completed_at = Clock::now();
    return;
  }
  // Lost 时没有跟踪 ROI/侧灯条，整图检测与姿态无关，可以先执行。
  // 活跃跟踪只有通过新鲜度/转速门限时才提前检测；最后仍用曝光姿态更新。
  const bool detect_before_pose = !tracker || !tracker->ready() ||
      tracker->state() == L3Estimation::TrackState::Lost;
  auto& image_pose = packet.image_pose;
  std::optional<cv::Rect> net_roi;
  const bool may_overlap = overlap && overlap->beginFrame();
  if (may_overlap && tracker && tracker->ready() &&
      tracker->state() == L3Estimation::TrackState::Tracking) {
    if (const auto provisional = serial.roiPosePreview(timestamp)) {
      net_roi = runtime::paddedPoseRoi(tracker->netFocusRoi(
        provisional, timestamp, frame.size(), armor_detector.net_aspect_ratio()), frame.size());
      frame_timing.pose_overlap = true;
    }
  }
  const auto resolve_pose = [&] {
    const auto started = Clock::now();
    if (!serial.waitPoseUntil(timestamp, pose_deadline)) {
      frame_timing.pose_timeout = true;
      L6Telemetry::logDebug("gimbal pose wait timeout");
    }
    frame_timing.pose_wait_ms = milliseconds(Clock::now(), started);
    image_pose = serial.gimbalPoseAt(timestamp);
  };
  const auto prepare_pose = [&] {
    resolve_pose();
    if (detect_before_pose || frame_timing.pose_overlap) return;
    net_roi = tracker->netFocusRoi(
      image_pose, timestamp, frame.size(), armor_detector.net_aspect_ratio());
  };
  const auto detect = [&](const std::optional<cv::Rect>& roi) {
    const auto started = Clock::now();
    auto result = armor_detector.detectFrame(frame, roi);
    frame_timing.detect_ms += milliseconds(Clock::now(), started);
    ++frame_timing.detection_passes;
    const auto stages = armor_detector.lastTiming();
    frame_timing.detection_stages.preprocess += stages.preprocess;
    frame_timing.detection_stages.infer += stages.infer;
    frame_timing.detection_stages.decode += stages.decode;
    frame_timing.detection_stages.refine += stages.refine;
    frame_timing.detection_stages.side_light += stages.side_light;
    return result;
  };
  auto& perception = packet.perception;
  perception = runtime::detectWithPose(
      !detect_before_pose && !frame_timing.pose_overlap, prepare_pose, [&] { return detect(net_roi); });
  const auto has_enemy = [&](const auto& armors) {
    return std::any_of(armors.begin(), armors.end(), [&](const auto& armor) {
      return isEnemyArmor(armor.color, state->enemy_color);
    });
  };
  if (frame_timing.pose_overlap) {
    const auto aligned_roi = tracker->netFocusRoi(
      image_pose, timestamp, frame.size(), armor_detector.net_aspect_ratio());
    // Validate the entire original ROI, not just detected corners (a missed plate
    // has no corners to check). No recursion: at most one corrected inference.
    const bool repair = !runtime::coversPoseRoi(*net_roi, aligned_roi) ||
      !has_enemy(perception.armors);
    frame_timing.pose_roi_repair = repair;
    if (repair) {
      net_roi = aligned_roi;
      perception = detect(net_roi);
    }
    overlap->finishAttempt(repair);
  }
  if (tracker && tracker->ready() && detect_before_pose) {
    auto candidates = perception.armors;
    std::erase_if(candidates, [&](const auto& armor) {
      return !isEnemyArmor(armor.color, state->enemy_color);
    });
    if (const auto init_roi = tracker->initRoi(
          candidates, image_pose, timestamp, frame.size(),
          armor_detector.net_aspect_ratio())) {
      const cv::Rect image_rect(0, 0, frame.cols, frame.rows);
      const cv::Rect first_roi = net_roi ? (*net_roi & image_rect) : image_rect;
      if ((*init_roi & image_rect) != first_roi) {
        auto refined = detect(*init_roi);
        if (has_enemy(refined.armors)) perception = std::move(refined);
      }
    }
  }
  // Defer pose-dependent side searches until the final inference is chosen.
  // Deduplicate against ALL classes/colors before filtering the enemy plates.
  if (tracker && tracker->ready()) {
    const auto started = Clock::now();
    armor_detector.completeSideLights(frame, perception,
      tracker->lightHints(image_pose, timestamp), enemyArmorColor(state->enemy_color));
    frame_timing.detect_ms += milliseconds(Clock::now(), started);
    frame_timing.detection_stages.side_light += armor_detector.lastTiming().side_light;
  }
  std::erase_if(perception.armors, [&](const auto& armor) {
    return !isEnemyArmor(armor.color, state->enemy_color);
  });

  // L3: 完整板与独立灯条联合关联，更新整车 IESKF。
  auto& target = packet.target;
  const auto estimate_started = Clock::now();
  if (tracker && tracker->ready()) {
    frame_timing.filter_input_age_ms = milliseconds(estimate_started, timestamp);
    target = tracker->track(
      perception.armors, perception.lights, image_pose, timestamp);
    frame_timing.filter_output_age_ms = milliseconds(Clock::now(), timestamp);
  }
  frame_timing.estimate_ms = milliseconds(Clock::now(), estimate_started);
  packet.track_state = tracker
    ? tracker->state()
    : L3Estimation::TrackState::Lost;

  packet.match_count = tracker ? tracker->lastMatchCount() : 0;
  packet.tracker_drops = tracker ? tracker->dropCount() : 0;
  packet.completed_at = Clock::now();
}

L2Perception::ArmorDetector makeArmorDetector(
  const runtime::AutoAimConfig& config)
{
  try {
    return runtime::makeDetector(config);
  } catch (const std::exception& error) {
    // 模型或 SDK 不可用时只在启动阶段记录一次；空 Detector 会持续返回安全的空结果。
    L6Telemetry::logError(
      "armor model unavailable",
      std::string{L2Perception::backendName(config.inference_backend)},
      config.model_path.string(), error.what());
    return {};
  }
}

/********************************** debug **********************************/
// 调试遥测的前向声明，定义在本文件末尾的 debug 段。放在那里是为了让 run()
// 的主流程从上往下读不被打断——遥测不参与瞄准链路，整段删掉也不影响自瞄。
nlohmann::json telemetryFrame(
  const std::optional<Eigen::Quaterniond>& q_world_barrel,
  const L1Sensor::RobotState& state,
  const std::vector<L3Estimation::Armor>& observations,
  const std::optional<L3Estimation::EskfTarget>& target,
  L3Estimation::TrackState track_state,
  const L4Planning::AimPlan& plan,
  const L5Control::FireDecision& fire,
  bool command_sent,
  const L1Sensor::SerialWorker& serial,
  std::chrono::steady_clock::time_point timestamp,
  int detect_count,
  std::uint64_t tracker_resets,
  std::uint64_t plan_rejects);
/********************************** debug **********************************/

} // namespace

namespace runtime {

AutoAimRuntime::AutoAimRuntime(const std::string &config_path)
    : config_path_(config_path) {}

void AutoAimRuntime::run() {
  running_ = true;
  AutoAimConfig auto_aim_config;
  L4Planning::PlannerTuning planner_tuning;
  try {
    auto_aim_config = loadAutoAimConfig("config/auto_aim.yaml");
    planner_tuning = L4Planning::loadPlannerTuning(
      "config/planner_config.yaml");
  } catch (const std::exception& error) {
    L6Telemetry::logError(
      "failed to load aiming configuration", error.what());
    running_ = false;
    return;
  }
  auto camera = std::make_shared<L1Sensor::Camera>(config_path_);
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    active_camera_ = camera;
  }
  // 启动时只加载一次模型；每帧仅执行预处理、推理和 Decoder。
  L2Perception::ArmorDetector armor_detector;
  std::unique_ptr<RuneRuntime> rune;
  if (auto_aim_config.mode == AimMode::Armor) {
    armor_detector = makeArmorDetector(auto_aim_config);
  } else {
    try {
      rune = std::make_unique<RuneRuntime>(auto_aim_config.rune, planner_tuning);
      const auto& detector = auto_aim_config.rune.detector;
      L6Telemetry::logInfo("rune detector loaded", detector.model_path.string(),
                          "device", detector.device, "confidence", detector.confidence_threshold,
                          "keypoint_confidence", detector.keypoint_confidence_threshold);
      const auto& geometry = auto_aim_config.rune.geometry;
      if (geometry.radius_m <= 0 || geometry.armor_half_width_m <= 0 ||
          geometry.armor_half_height_m <= 0) {
        L6Telemetry::logWarn(
          "Rune detection enabled; aiming requires measured rune.geometry in config/auto_aim.yaml");
      }
    } catch (const std::exception& error) {
      L6Telemetry::logError("rune pipeline unavailable", error.what());
    }
  }

  //配置并启动串口
  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    L6Telemetry::logWarn("Failed to start serial worker.");
  }
  // L3 EskfTracker 持有初始化 PnP 与整车 IESKF。标定缺失时继续运行检测，
  // 但后续不得生成有效瞄准/开火命令。
  std::optional<L3Estimation::EskfTracker> tracker;
  const auto& camera_calibration = camera->calibration();
  if (!camera_calibration) {
    L6Telemetry::logWarn("Tracker disabled: camera calibration is missing");
  } else if (auto_aim_config.mode == AimMode::Armor) {
    tracker.emplace(
      *camera_calibration,
      auto_aim_config.armor,
      auto_aim_config.ieskf_tracker,
      auto_aim_config.ieskf_target);
    if (tracker->ready()) {
      L6Telemetry::logInfo("L3 tracker configured");
    } else {
      L6Telemetry::logWarn(
        "Tracker disabled: calibration, armor, or tracker config is invalid");
    }
  }

  L4Planning::Planner planner(planner_tuning.planner);
  L4Planning::PlannerContext planner_context;
  planner_context.config = planner_tuning.planner;
  planner_context.latency = planner_tuning.latency;
  planner_context.armor_score_weights = planner_tuning.armor_score_weights;
  planner_context.facing_angle_good = planner_tuning.facing_angle_good;
  planner_context.facing_angle_bad = planner_tuning.facing_angle_bad;
  L5Control::Controller controller(
    auto_aim_config.fire,
    auto_aim_config.runtime.command_jump_threshold);
  L6Telemetry::logInfo(
    "armor fire gates: stop_fire_on_temp_lost", auto_aim_config.fire.stop_fire_on_temp_lost,
    "stop_fire_on_aim_error", auto_aim_config.fire.stop_fire_on_aim_error);

  FrameTiming frame_timing;
  // 规划失败时不能只是“不更新命令”：上一条 shoot=true 在
  // command_timeout 内仍可能被重复发送。这里保留最后角度并立即关火。
  std::optional<L1Sensor::SerialCommand> last_command;
  const auto sendCommand = [&](const L1Sensor::SerialCommand& command,
                               bool single_shoot = false) {
    if (serial_started) {
      serial.updateCommand(command, single_shoot);
      frame_timing.queued_at = Clock::now();
      last_command = command;
    }
  };
  const auto sendSafeHold = [&]() {
    if (!last_command) return;
    auto command = *last_command;
    command.shoot = false;
    command.yaw_velocity = command.pitch_velocity = 0;
    command.yaw_acceleration = command.pitch_acceleration = 0;
    sendCommand(command);
  };

  const auto stopAimSession = [&]() {
    // Tracker is owned by the vision producer; reset it there or after join.
    planner.resetTracking();
    if (rune) rune->suspend();
    sendSafeHold();
  };

  /******************************** debug *********************************/
  // 叠加层默认关闭：imshow 的耗时会计进 image_to_plan，而且比赛用的机器
  // 没有显示器，无条件 namedWindow 会直接抛。
  const bool overlay_enabled = auto_aim_config.debug.overlay;
  if (overlay_enabled) {
    cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);
  }
  std::uint64_t frame_index = 0;
  bool save_rune_frame = false;
  Clock::time_point last_rune_diagnostic_log{};
  Clock::time_point last_armor_detection_only_log{};
  // 叠加层要把世界系位姿投回图像，需要一个求解器。这里另建一个与 Tracker
  // 内部同参数的实例，只做重投影、不参与滤波——不为了画图去开 Tracker 的内部。
  std::optional<L3Estimation::PnpSolver> overlay_solver;
  if (overlay_enabled && camera_calibration) {
    overlay_solver.emplace(*camera_calibration, auto_aim_config.armor);
  }

  // 曲线遥测与叠加层各自独立开关：实车上标定延迟链时没有显示器，需要的
  // 恰好是曲线而不是画面。UDP 是无连接的，没人接收也不会阻塞或报错。
  std::optional<L6Telemetry::UdpJsonSender> plotter;
  if (auto_aim_config.debug.plot) {
    plotter.emplace(
      auto_aim_config.debug.plot_host,
      static_cast<std::uint16_t>(auto_aim_config.debug.plot_port));
    L6Telemetry::logInfo(
      "telemetry enabled", plotter->host(), std::to_string(plotter->port()));
  }
  const auto selected_mode = workMode(auto_aim_config.mode);
  L6Telemetry::logInfo("aiming mode selected by config/auto_aim.yaml:",
                      static_cast<int>(auto_aim_config.mode), L1Sensor::toString(selected_mode));
  std::uint64_t plan_reject_count = 0;
  bool using_default_bullet_speed = false;
  Clock::time_point last_pitch_diagnostic_log{};
  Clock::time_point last_fire_diagnostic_log{};
  std::uint64_t fire_diagnostic_frames = 0;
  std::uint64_t fire_allowed_frames = 0;
  std::uint64_t fire_unlocked_frames = 0;
  std::uint64_t fire_outside_window_frames = 0;
  std::array<std::uint64_t,
    static_cast<std::size_t>(L5Control::RejectReason::NonFinite) + 1> fire_reject_frames{};
  Clock::time_point fire_stats_started = Clock::now();
  std::uint64_t fire_stats_serial_tx = serial.sentCommandCount();
  /******************************** debug *********************************/

  // This producer is also used by synchronous mode, keeping the algorithms identical.
  // In pipeline mode only the worker may access detector/tracker after startup.
  const bool armor_model_ready = armor_detector.ready();
  runtime::PoseOverlapState pose_overlap_state;
  const auto nextArmorFrame = [&]() -> std::optional<ArmorVisionPacket> {
    if (!running_) return std::nullopt;
    ArmorVisionPacket packet;
    auto& frame = packet.image;
    auto& frame_timing = packet.timing;
    auto& timestamp = frame_timing.captured_at;
    if (!camera->read(frame, timestamp)) {
      if (!running_) return std::nullopt;
      return packet; // Explicit capture timeout; consumer immediately stops firing.
    }
    frame_timing.ready_at = Clock::now();
    packet.state = serial.latestState();
    processArmorVision(packet, armor_detector, tracker, serial, serial_started,
                       serial_config.pose_wait_ms,
                       auto_aim_config.runtime.pose_detection_overlap ? &pose_overlap_state : nullptr);
    return packet;
  };

  using ArmorPipeline = SingleFramePipeline<ArmorVisionPacket>;
  std::unique_ptr<ArmorPipeline> armor_pipeline;
  if (auto_aim_config.mode == AimMode::Armor && auto_aim_config.runtime.armor_pipeline) {
    armor_pipeline = std::make_unique<ArmorPipeline>(nextArmorFrame);
  }
  std::uint64_t stale_results = 0;

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;
  // Capture stays latest-only. The handoff permits at most one processed frame
  // ahead of planning/display; no FIFO of historical camera images is introduced.
  try {
    while (running_) {
      frame_timing = {};
      std::optional<ArmorVisionPacket> armor_packet;
      if (auto_aim_config.mode == AimMode::Armor) {
        if (armor_pipeline) {
          ArmorVisionPacket packet;
          const auto status = armor_pipeline->readFor(packet, std::chrono::milliseconds(50));
          if (status != ArmorPipeline::ReadStatus::Value) {
            sendSafeHold();
            if (status == ArmorPipeline::ReadStatus::Closed || !running_) break;
            continue;
          }
          armor_packet = std::move(packet);
        } else {
          armor_packet = nextArmorFrame();
        }
        if (!running_ || !armor_packet) {
          sendSafeHold();
          break;
        }
        if (armor_packet->image.empty()) {
          sendSafeHold();
          continue;
        }
        frame = std::move(armor_packet->image);
        frame_timing = armor_packet->timing;
        timestamp = frame_timing.captured_at;
        frame_timing.result_wait_ms = milliseconds(Clock::now(), armor_packet->completed_at);
        if (armor_pipeline && milliseconds(Clock::now(), timestamp) >
              auto_aim_config.runtime.pipeline_max_result_age_ms) {
          ++stale_results;
          frame_timing.stale_results = stale_results;
          sendSafeHold();
          if (plotter) {
            (void)plotter->send(nlohmann::json{
              {"pipeline", pipelineTiming(frame_timing, serial)}, {"result_rejected", true}});
          }
          continue;
        }
        frame_timing.stale_results = stale_results;
      } else {
        if (!camera->read(frame, timestamp)) {
          sendSafeHold();
          if (!running_) break;
          continue;
        }
        frame_timing.captured_at = timestamp;
        frame_timing.ready_at = Clock::now();
      }
      const auto pose_deadline = frame_timing.ready_at +
          std::chrono::milliseconds(serial_config.pose_wait_ms);
      const auto state = armor_packet ? armor_packet->state : serial.latestState();
      // 图像检测不依赖串口。缺少反馈时只做整图检测，保持停瞄，
      // 不构造虚假的姿态、敌方颜色或跟踪目标。
      if (auto_aim_config.mode == AimMode::Armor && (!serial_started || !state)) {
        stopAimSession();
        const auto& perception = armor_packet->perception;
        if (Clock::now() - last_armor_detection_only_log >= std::chrono::seconds(2)) {
          L6Telemetry::logWarn("armor detection only: serial feedback unavailable; detections",
                              perception.armors.size());
          last_armor_detection_only_log = Clock::now();
        }
        if (overlay_enabled && frame_index % auto_aim_config.debug.overlay_every == 0) {
          // 没有敌方颜色时显示所有网络检出，仅供查看，不进入跟踪与规划。
          L6Telemetry::drawArmorDetections(frame, perception.armors);
          L6Telemetry::drawOutlinedText(
            frame, armor_model_ready ? "Detection only: no serial feedback"
                                          : "Armor model unavailable; see log",
            {20, 40}, {0, 165, 255});
        }
      } else {
        const L1Sensor::WorkMode mode = selected_mode;
        switch (mode) {
          case L1Sensor::WorkMode::AutoAim:
          case L1Sensor::WorkMode::Outpost:
          case L1Sensor::WorkMode::Idle: {
            const bool aiming = mode != L1Sensor::WorkMode::Idle;
            const auto& image_pose = armor_packet->image_pose;
            const auto& perception = armor_packet->perception;
            const auto& target = armor_packet->target;
            const auto track_state = armor_packet->track_state;

            L4Planning::AimPlan plan;
            std::optional<L5Control::SerialCommand> command;
            if (aiming) {
              // L4/L5 保留 Bruce0178 分支现有实现：动态窗口选板、弹道迭代与
              // 开火闸门均不被目标分支覆盖。
              const auto plan_time = std::chrono::steady_clock::now();
              const auto actual_pose = serial.latestGimbalPose();
              planner_context.planning_time = plan_time;
              auto planning_state = *state;
              planning_state.timestamp = plan_time;
              const bool mcu_bullet_speed_valid =
                std::isfinite(state->bullet_speed) && state->bullet_speed > 0.0;
              if (!mcu_bullet_speed_valid) {
                planning_state.bullet_speed = planner_tuning.default_bullet_speed;
                if (!using_default_bullet_speed) {
                  L6Telemetry::logWarn(
                    "invalid MCU bullet speed", state->bullet_speed,
                    "; using configured fallback for tracking and firing",
                    planner_tuning.default_bullet_speed, "m/s");
                }
              } else if (using_default_bullet_speed) {
                L6Telemetry::logInfo(
                  "MCU bullet speed recovered", state->bullet_speed, "m/s");
              }
              using_default_bullet_speed = !mcu_bullet_speed_valid;
              const bool effective_bullet_speed_valid =
                std::isfinite(planning_state.bullet_speed) &&
                planning_state.bullet_speed > 0.0;
              plan = planner.plan(
                toL4TargetState(target), planning_state, planner_context);
              if (!plan.valid) {
                ++plan_reject_count;
              }
              command = controller.update(
                target, track_state, plan, actual_pose,
                effective_bullet_speed_valid);
              if (command) {
                sendCommand(*command);
              }
              frame_timing.plan_control_ms = milliseconds(Clock::now(), plan_time);

              // 分开记录几何角、弹道和最终命令，避免把抬枪过高直接归因于弹道。
              // 角度统一为向上正；queued 包含串口偏置，feedback 来自最新枪口姿态。
              if ((overlay_enabled || plotter.has_value()) && plan.valid && command
                  && Clock::now() - last_pitch_diagnostic_log >= std::chrono::seconds(1)) {
                last_pitch_diagnostic_log = Clock::now();
                constexpr double kRadToDeg = 180.0 / std::numbers::pi;
                const auto line_of_sight = [](const Eigen::Vector3d& point) {
                  return std::atan2(point.z(), std::hypot(point.x(), point.y()));
                };
                const double los_pitch = line_of_sight(plan.aim_point_barrel);
                const double feedback_pitch = actual_pose
                  ? -L6Telemetry::eulers(actual_pose->toRotationMatrix(), 2, 1, 0)[1]
                  : std::numeric_limits<double>::quiet_NaN();
                L6Telemetry::logInfo(
                  "pitch diagnostic (deg, up+): range_m", plan.aim_point_barrel.norm(),
                  "height_m", plan.aim_point_barrel.z(),
                  "center_los", line_of_sight(plan.armor_center_barrel) * kRadToDeg,
                  "aim_los", los_pitch * kRadToDeg,
                  "ballistic_add", (plan.pitch - los_pitch) * kRadToDeg,
                  "reference", plan.pitch * kRadToDeg,
                  "queued", -command->pitch * kRadToDeg - serial_config.command_pitch_offset_deg,
                  "feedback", feedback_pitch * kRadToDeg,
                  "speed_mps", plan.bullet_speed,
                  "mcu_speed_mps", state->bullet_speed,
                  "drag_per_s", plan.drag_coefficient,
                  "mpc", plan.using_MPC);
              }
            } else {
              // Idle 继续预热识别与预测，但不运行选板、不输出开火命令。
              planner.resetTracking();
              sendSafeHold();
            }

            /*************************** debug ****************************/
            // 本帧命令已入队；这些工作仍会占用下一帧的处理时间。
            if (aiming && (overlay_enabled || plotter.has_value())) {
              const auto& fire = controller.lastDecision();
              ++fire_diagnostic_frames;
              fire_allowed_frames += fire.shoot ? 1 : 0;
              fire_unlocked_frames += !plan.tracked_ready ? 1 : 0;
              fire_outside_window_frames += !plan.within_firing_window ? 1 : 0;
              for (const auto reason : fire.reasons) {
                ++fire_reject_frames[static_cast<std::size_t>(reason)];
              }
            }
            if (aiming && (overlay_enabled || plotter.has_value())
                && Clock::now() - last_fire_diagnostic_log >= std::chrono::seconds(1)) {
              last_fire_diagnostic_log = Clock::now();
              const auto& fire = controller.lastDecision();
              std::string reasons;
              for (const auto reason : fire.reasons) {
                if (!reasons.empty()) reasons += ',';
                reasons += L5Control::toString(reason);
              }
              constexpr double kRadToDeg = 180.0 / std::numbers::pi;
              const auto sent = serial.latestSentCommand();
              L6Telemetry::logInfo(
                "fire diagnostic: shoot", fire.shoot,
                "last_tx_shoot", sent ? static_cast<int>(sent->shoot) : -1,
                "reasons", reasons.empty() ? "none" : reasons,
                "plan_valid", plan.valid, "lock", plan.tracked_ready,
                "window", plan.within_firing_window,
                "aim_ok", fire.aim_error_within_tolerance,
                "plate_hit_diag", fire.hit_inside_plate,
                "yaw_err_deg", fire.yaw_error * kRadToDeg,
                "pitch_err_deg", fire.pitch_error * kRadToDeg,
                "yaw_tol_deg", fire.tolerance.yaw * kRadToDeg,
                "pitch_tol_deg", fire.tolerance.pitch * kRadToDeg,
                "aim_u_m", plan.in_plate_aim.offset.x(),
                "aim_v_m", plan.in_plate_aim.offset.y(),
                "hit_u_m", fire.predicted_hit_uv.x(),
                "hit_v_m", fire.predicted_hit_uv.y(),
                "hit_t_s", fire.predicted_fly_time);

              // 每秒汇总全部帧，避免一秒一条瞬时 shoot 掩盖频繁断火。
              // 一帧可能被多项门控拒绝，各原因的计数允许重叠。
              std::string reject_counts;
              for (std::size_t index = 0; index < fire_reject_frames.size(); ++index) {
                if (fire_reject_frames[index] == 0) continue;
                if (!reject_counts.empty()) reject_counts += ',';
                reject_counts += L5Control::toString(static_cast<L5Control::RejectReason>(index))
                  + ':' + std::to_string(fire_reject_frames[index]);
              }
              const auto stats_time = Clock::now();
              const double elapsed = std::chrono::duration<double>(stats_time - fire_stats_started).count();
              const auto tx_count = serial.sentCommandCount();
              const double percent_per_frame = 100.0 / fire_diagnostic_frames;
              L6Telemetry::logInfo(
                "fire gate stats: frames", fire_diagnostic_frames,
                "fps", fire_diagnostic_frames / elapsed,
                "allow_pct", fire_allowed_frames * percent_per_frame,
                "unlocked_pct", fire_unlocked_frames * percent_per_frame,
                "outside_window_pct", fire_outside_window_frames * percent_per_frame,
                "reject_counts", reject_counts.empty() ? "none" : reject_counts,
                "serial_tx_hz", (tx_count - fire_stats_serial_tx) / elapsed);
              fire_diagnostic_frames = fire_allowed_frames = 0;
              fire_unlocked_frames = fire_outside_window_frames = 0;
              fire_reject_frames.fill(0);
              fire_stats_started = stats_time;
              fire_stats_serial_tx = tx_count;
            }
            if (plotter) {
              static const std::vector<L3Estimation::Armor> kNoObservations;
              auto telemetry = telemetryFrame(
                image_pose, *state, kNoObservations, target, track_state, plan,
                controller.lastDecision(), command.has_value(), serial, timestamp,
                armor_packet->match_count, armor_packet->tracker_drops, plan_reject_count);
              telemetry["pipeline"] = pipelineTiming(frame_timing, serial);
              (void)plotter->send(telemetry);
            }
            if (overlay_solver && armor_packet->has_tracker &&
                frame_index % auto_aim_config.debug.overlay_every == 0) {
              overlay_solver->set_R_world_barrel(image_pose);
              L6Telemetry::drawAimOverlay(
                frame,
                {.detections = perception.armors,
                 .observations = {},
                 .target = target,
                 .track_state = track_state,
                 .plan = plan,
                 .fire = controller.lastDecision(),
                 .q_world_barrel = image_pose},
                *overlay_solver, *camera_calibration);
            }
            /*************************** debug ****************************/
            break;
          }

          case L1Sensor::WorkMode::SmallBuff:
          case L1Sensor::WorkMode::BigBuff: {
            if (!rune) {
              sendSafeHold();
              if (overlay_enabled) {
                cv::putText(frame, "Rune detector unavailable; see log", {20, 40},
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
              }
              break;
            }
            if (save_rune_frame) {
              save_rune_frame = false;
              try {
                const auto path = std::filesystem::path("logs/rune_input.png");
                std::filesystem::create_directories(path.parent_path());
                if (!cv::imwrite(path.string(), frame))
                  throw std::runtime_error("imwrite returned false");
                L6Telemetry::logInfo("saved raw rune frame", path.string());
              } catch (const std::exception& error) {
                L6Telemetry::logError("failed to save raw rune frame", error.what());
              }
            }
            Rune::Frame input;
            input.bgr = frame;
            input.captured_at = timestamp;
            input.mode = mode == L1Sensor::WorkMode::SmallBuff
                ? Rune::Mode::Small : Rune::Mode::Big;
            input.color = state && state->enemy_color==L1Sensor::EnemyColor::Blue
                ? Rune::Color::Blue : Rune::Color::Red;
            if (camera_calibration) {
              (void)Rune::applyCameraCalibration(input, *camera_calibration);
            }
            try {
              // 检测只依赖图像：先推理，让串口接收线程同时补齐曝光后的姿态包。
              const auto detect_started = Clock::now();
              rune->detect(input);
              frame_timing.detect_ms = milliseconds(Clock::now(), detect_started);
              frame_timing.detection_passes = 1;
              if ((overlay_enabled || plotter) &&
                  (last_rune_diagnostic_log == Clock::time_point{} ||
                   Clock::now() - last_rune_diagnostic_log >= std::chrono::seconds(1))) {
                if (const auto diagnostics = rune->detectorDiagnostics()) {
                  L6Telemetry::logDebug("rune detection", Rune::formatDetectionDiagnostics(*diagnostics),
                                        "image", frame.cols, frame.rows);
                }
                last_rune_diagnostic_log = Clock::now();
              }
              const auto pose_wait_started = Clock::now();
              (void)serial.waitPoseUntil(timestamp, pose_deadline);
              frame_timing.pose_wait_ms = milliseconds(Clock::now(), pose_wait_started);
              input.q_world_barrel = serial.gimbalPoseAt(timestamp);
              // 推理耗时计入图像年龄；规划及反馈使用推理完成后的时刻。
              const auto feedback = serial.latestState();
              const auto actual_pose = serial.latestGimbalPose();
              const auto command_at = std::chrono::steady_clock::now();
              if (feedback) {
                if (auto command = rune->process(input, *feedback, actual_pose, command_at)) {
                  sendCommand(*command, true);
                } else {
                  sendSafeHold();
                }
              } else {
                rune->suspend();
                sendSafeHold();
              }
              // Rune::process 包含轮廓、PnP、估计、规划及控制。
              frame_timing.plan_control_ms = milliseconds(Clock::now(), command_at);
            } catch (const std::exception& error) {
              L6Telemetry::logError("rune frame processing failed", error.what());
              rune->suspend();
              sendSafeHold();
            }
            if (plotter) {
              nlohmann::json telemetry{
                  {"t", std::chrono::duration<double>(timestamp.time_since_epoch()).count()},
                  {"mode", static_cast<int>(auto_aim_config.mode)},
                  {"pipeline", pipelineTiming(frame_timing, serial)}};
              (void)plotter->send(telemetry);
            }
            if (overlay_enabled && frame_index % auto_aim_config.debug.overlay_every == 0) {
              Rune::drawDetectionOverlay(frame, input.detections, input.mode);
              if (rune->lastMotion())
                Rune::drawTrackingOverlay(frame, input, rune->lastMotion()->pose);
              if (const auto diagnostics = rune->detectorDiagnostics())
                Rune::drawDetectionDiagnostics(frame, *diagnostics);
              const auto& command = rune->lastCommand();
              const std::string label = (mode == L1Sensor::WorkMode::SmallBuff ? "Small rune" : "Big rune") +
                  std::string(" status=") + std::to_string(static_cast<int>(command.status)) +
                  ((!serial_started || !state) ? " NO SERIAL FEEDBACK" : "") +
                  (command.using_mpc ? " MPC" : "") + (command.fire ? " FIRE" : "");
              cv::putText(frame, label, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
            }
            break;
          }

          default:
            stopAimSession();
            break;
        }
      }

      ++frame_index;
      /******************************* debug ********************************/
      if (overlay_enabled) {
        L6Telemetry::drawImageCenter(frame);
        cv::imshow("auto_aim", frame);
        const int key = cv::waitKey(1);
        if ((key == 's' || key == 'S') && auto_aim_config.mode != AimMode::Armor)
          save_rune_frame = true;
        if (key == 27 || key == 'q' || key == 'Q') {
          running_ = false;
        }
      }
      /******************************* debug ********************************/
    }

  } catch (const std::exception& error) {
    L6Telemetry::logError("aim runtime failed", error.what());
  } catch (...) {
    L6Telemetry::logError("aim runtime failed with unknown exception");
  }

  running_ = false;
  sendSafeHold();
  // Join before resetting tracker or stopping its camera/serial dependencies.
  armor_pipeline.reset();
  if (tracker) tracker->reset();
  stopAimSession();
  serial.stop();
  camera->stop();
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    if (active_camera_ == camera) {
      active_camera_.reset();
    }
  }
  if (overlay_enabled) {
    cv::destroyWindow("auto_aim");
  }
}

void AutoAimRuntime::stop() {
  running_ = false;
  std::shared_ptr<L1Sensor::Camera> camera;
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    camera = active_camera_;
  }
  if (camera) {
    camera->stop();
  }
}

} // namespace runtime

/********************************** debug **********************************/
// 以下只服务调试，不参与瞄准链路。debug.plot 关闭时一次都不会被调用。
namespace {

// PlotJuggler 遥测。**分节点是硬要求**：实测云台姿态 gimbal/ 和规划出的云台
// 姿态 aim/ 必须落在曲线树的不同分支上，否则"跟随误差"这类靠两条曲线相减
// 看出来的量根本没法读；同理观测 obs/ 与滤波结果 ekf/ 也不能混在一层。
// 角度统一转成 degree、时间统一转成 ms——曲线是给人看的，不是给代码读的。
nlohmann::json telemetryFrame(
  const std::optional<Eigen::Quaterniond>& q_world_barrel,
  const L1Sensor::RobotState& state,
  const std::vector<L3Estimation::Armor>& observations,
  const std::optional<L3Estimation::EskfTarget>& target,
  L3Estimation::TrackState track_state,
  const L4Planning::AimPlan& plan,
  const L5Control::FireDecision& fire,
  bool command_sent,
  const L1Sensor::SerialWorker& serial,
  std::chrono::steady_clock::time_point timestamp,
  int detect_count,
  std::uint64_t tracker_resets,
  std::uint64_t plan_rejects)
{
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  nlohmann::json data;

  data["t"] = std::chrono::duration<double>(timestamp.time_since_epoch()).count();

  // gimbal: L1 实测的云台姿态与弹速。
  if (q_world_barrel) {
    const Eigen::Vector3d ypr =
      L6Telemetry::eulers(q_world_barrel->toRotationMatrix(), 2, 1, 0);
    data["gimbal"]["yaw"] = ypr[0] * kRadToDeg;
    data["gimbal"]["pitch"] = ypr[1] * kRadToDeg;
  }
  data["gimbal"]["bullet_speed"] = state.bullet_speed;
  data["gimbal"]["heat"] = state.heat;

  data["serial"]["rx"] = serial.receivedStateCount();
  data["serial"]["rx_dropped"] = serial.droppedPacketCount();
  data["serial"]["rx_skipped_bytes"] = serial.skippedByteCount();
  data["serial"]["tx"] = serial.sentCommandCount();
  data["serial"]["tx_failed"] = serial.failedCommandCount();
  data["serial"]["pose_before_history"] = serial.poseBeforeHistoryCount();
  data["serial"]["pose_after_history"] = serial.poseAfterHistoryCount();

  // track: 状态机与本帧真正进滤波器的观测数量。
  data["track"]["state"] = static_cast<int>(track_state);
  data["track"]["n_obs"] = static_cast<int>(observations.size());
  data["track"]["match_count"] = detect_count;
  data["track"]["drops"] = tracker_resets;
  data["aim"]["rejects"] = plan_rejects;

  // obs: 单板 PnP 的原始观测。固定取图像最左的一块——多板时若按检测顺序取，
  // 曲线会在两块板之间来回跳，看不出任何趋势。
  const auto leftmost = std::min_element(
    observations.begin(), observations.end(),
    [](const L3Estimation::Armor& a, const L3Estimation::Armor& b) {
      return a.center.x < b.center.x;
    });
  if (leftmost != observations.end()) {
    data["obs"]["x"] = leftmost->xyz_in_world[0];
    data["obs"]["y"] = leftmost->xyz_in_world[1];
    data["obs"]["z"] = leftmost->xyz_in_world[2];
    data["obs"]["distance"] = leftmost->xyz_in_world.norm();
    // yaw 是选解后的、yaw_raw 是单次 PnP 的原始解。两条一起画才看得出
    // optimize_yaw 在哪些帧救了场、哪些帧把解带偏了。
    data["obs"]["yaw"] = leftmost->ypr_in_world[0] * kRadToDeg;
    data["obs"]["yaw_raw"] = leftmost->yaw_raw * kRadToDeg;
    data["obs"]["reproj_err"] = leftmost->reprojection_error;
  }

  // ekf: 目标提交的十三维误差状态整车模型。
  if (target) {
    const Eigen::VectorXd x = target->ekf_x();
    data["ekf"]["x"] = x[0];
    data["ekf"]["vx"] = x[1];
    data["ekf"]["y"] = x[2];
    data["ekf"]["vy"] = x[3];
    data["ekf"]["z"] = x[4];
    data["ekf"]["vz"] = x[5];
    data["ekf"]["a"] = x[6] * kRadToDeg;
    data["ekf"]["w"] = x[7];
    data["ekf"]["r"] = x[8];
    data["ekf"]["r2_or_dz1"] = x[9];
    data["ekf"]["height_or_dz2"] = x[10];
    data["ekf"]["rot_y"] = target->rawState()[11] * kRadToDeg;
    data["ekf"]["rot_x"] = target->rawState()[12] * kRadToDeg;
    data["ekf"]["last_id"] = target->last_id;
    data["ekf"]["jumped"] = target->jumped ? 1 : 0;

    const auto& residual = target->lastLightResidual();
    data["ekf"]["nis"] = target->lastNis();
    data["ekf"]["nis_dof"] = target->lastNisDof();
    const auto& innovation = target->lastInnovation();
    const bool updated = innovation.timestamp == timestamp;
    data["ekf"]["innovation_updated"] = updated;
    data["ekf"]["matched_armors"] = updated ? innovation.armors : 0;
    data["ekf"]["isolated_lights"] = updated ? innovation.isolated_lights : 0;
    data["ekf"]["corner_innovations"] = nlohmann::json::array();
    if (updated) {
      for (const auto& corner : innovation.corners)
        data["ekf"]["corner_innovations"].push_back({
          {"armor_id", corner.armor_id}, {"is_left", corner.is_left},
          {"isolated", corner.isolated}, {"endpoint", corner.endpoint},
          {"du_px", corner.du}, {"dv_px", corner.dv},
          {"raw_nis", corner.nis}, {"variance_scale", corner.variance_scale}});
    }

    data["ekf"]["lights"] = residual.light_count;
    data["ekf"]["res_shift_perp"] = residual.shift_perp.mean;
    data["ekf"]["res_shift_along"] = residual.shift_along.mean;
    data["ekf"]["res_tilt"] = residual.tilt.mean;
    data["ekf"]["res_length"] = residual.length.mean;
  }

  // aim: L4 规划出的云台目标姿态。**与 gimbal/ 分开**，两者同图即跟随误差。
  data["aim"]["valid"] = plan.valid ? 1 : 0;
  data["aim"]["tracked_phase"] = static_cast<int>(plan.tracked_phase);
  data["aim"]["tracked_ready"] = plan.tracked_ready ? 1 : 0;
  data["aim"]["within_firing_window"] =
    plan.within_firing_window ? 1 : 0;
  data["aim"]["fire_permitted"] = plan.fire_permitted ? 1 : 0;
  if (plan.valid) {
    const double command_yaw = plan.using_MPC && !plan.samples.empty()
      ? plan.samples.front().yaw
      : plan.yaw;
    const double command_pitch = plan.using_MPC && !plan.samples.empty()
      ? plan.samples.front().pitch
      : plan.pitch;
    data["aim"]["yaw"] = command_yaw * kRadToDeg;
    data["aim"]["pitch"] = command_pitch * kRadToDeg;
  }
  data["aim"]["armor_id"] = plan.armor_id;

  // delay: 五段延迟链。绝不合并成一个标量——上车标定 send_to_control 时
  // 要能看出是哪一段在变。
  double image_to_plan = 0.0;
  double before_fire = 0.0;
  if (target && plan.generated_at >= target->t()) {
    image_to_plan = std::chrono::duration<double>(
      plan.generated_at - target->t()).count();
  }
  if (target && plan.valid && plan.impact_time >= target->t()) {
    const auto fire_time = plan.impact_time -
      std::chrono::duration_cast<L4Planning::TimePoint::duration>(
        std::chrono::duration<double>(plan.fly_time));
    before_fire = std::chrono::duration<double>(
      fire_time - target->t()).count();
  }
  data["delay"]["image_to_plan"] = image_to_plan * 1e3;
  data["delay"]["before_fire"] = before_fire * 1e3;
  data["delay"]["fire_to_hit"] = plan.fly_time * 1e3;

  // fire: 可行性与实际下发分开。feasible=1 而 shoot=0 就是被 shoot_enable
  // 或跳变检查拦下来了，reason 给出第一条原因（-1 表示无拒绝）。
  data["fire"]["feasible"] = fire.fire_feasible ? 1 : 0;
  data["fire"]["shoot"] = fire.shoot ? 1 : 0;
  data["fire"]["sent"] = command_sent ? 1 : 0;
  data["fire"]["yaw_err"] = fire.yaw_error * kRadToDeg;
  data["fire"]["pitch_err"] = fire.pitch_error * kRadToDeg;
  data["fire"]["aim_error_within_tolerance"] = fire.aim_error_within_tolerance ? 1 : 0;
  data["fire"]["hit_inside_plate"] = fire.hit_inside_plate ? 1 : 0;
  data["fire"]["hit_u_m"] = fire.predicted_hit_uv.x();
  data["fire"]["hit_v_m"] = fire.predicted_hit_uv.y();
  data["fire"]["hit_fly_time_ms"] = fire.predicted_fly_time * 1e3;
  data["fire"]["tol_yaw"] = fire.tolerance.yaw * kRadToDeg;
  data["fire"]["tol_pitch"] = fire.tolerance.pitch * kRadToDeg;
  data["fire"]["reason"] =
    fire.reasons.empty() ? -1 : static_cast<int>(fire.reasons.front());
  data["fire"]["reason_count"] = static_cast<int>(fire.reasons.size());
  for (std::size_t index = 0; index < fire.reasons.size(); ++index) {
    data["fire"]["reason_" + std::to_string(index)] =
      static_cast<int>(fire.reasons[index]);
  }

  return data;
}

}  // namespace
/********************************** debug **********************************/

int main()
{
  L6Telemetry::initLogger();
  runtime::AutoAimRuntime runtime("config/camera_config.yaml");
  runtime.run();
  L6Telemetry::flushLogger();
  return 0;
}
