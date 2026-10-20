#include "runtime/auto_aim_runtime.hpp"
#include "l1_sensor/buff/transform.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/armor/tracker.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l5_control/controller.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/buff/overlay.hpp"
#include "l6_telemetry/fps_counter.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "l6_telemetry/udp_json_sender.hpp"
#include "l6_telemetry/web_cmd.hpp"
#include "l6_telemetry/web_sink.hpp"
#include "runtime/auto_aim_config.hpp"
#include "runtime/buff/buff_config.hpp"
#include "runtime/buff/buff_runtime.hpp"
#include "runtime/runtime_io.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace {

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

struct TalosMetrics {
  std::uint64_t frames{};
  std::uint64_t observations{};
  std::uint64_t matches{};
  double error_sum{};
};

// GroundTruth 只用于验收统计，结果绝不回灌检测、跟踪或规划。
void updateTalosMetrics(
  TalosMetrics& metrics,
  const std::vector<L3Estimation::Armor>& observations,
  const L1Sensor::talos::GroundTruthBatch& truth,
  L1Sensor::EnemyColor enemy)
{
  const std::uint8_t team = enemy == L1Sensor::EnemyColor::Red ? 0 : 1;
  constexpr double kMatchGateMeters = 0.6;
  ++metrics.frames;
  for (const auto& observation : observations) {
    if (!observation.xyz_in_world.allFinite()) continue;
    ++metrics.observations;
    double best = kMatchGateMeters;
    for (std::uint32_t index = 0;
         index < truth.target_count && index < L1Sensor::talos::kMaxGroundTruthTargets;
         ++index) {
      const auto& target = truth.targets[index];
      if (target.team != team) continue;
      const Eigen::Vector3d position{
        target.position[0], target.position[1], target.position[2]};
      best = std::min(best, (observation.xyz_in_world - position).norm());
    }
    if (best < kMatchGateMeters) {
      ++metrics.matches;
      metrics.error_sum += best;
    }
  }
  if (metrics.frames % 120 == 0) {
    const double precision = metrics.observations == 0 ? 0.0
      : static_cast<double>(metrics.matches) / metrics.observations;
    const double mean_error = metrics.matches == 0 ? 0.0
      : metrics.error_sum / metrics.matches;
    L6Telemetry::logInfo(
      "Talos GT armor metrics", "frames", metrics.frames,
      "observations", metrics.observations, "matched", metrics.matches,
      "precision", precision, "mean_error_m", mean_error);
  }
}

// 把配置里的字符串解析成 WorkMode。空串或无法识别都返回空，表示不覆盖——
// 拼错模式名不该悄悄退化成 Idle 或 AutoAim，那两种误判的后果完全不同。
std::optional<L1Sensor::WorkMode> parseWorkMode(const std::string& name)
{
  if (name.empty()) {
    return std::nullopt;
  }
  if (name == "auto_aim") return L1Sensor::WorkMode::AutoAim;
  if (name == "outpost") return L1Sensor::WorkMode::Outpost;
  if (name == "small_buff") return L1Sensor::WorkMode::SmallBuff;
  if (name == "big_buff") return L1Sensor::WorkMode::BigBuff;
  if (name == "idle") return L1Sensor::WorkMode::Idle;
  L6Telemetry::logError(
    "debug.force_work_mode is not a known mode, ignored:", name,
    "| valid: auto_aim outpost small_buff big_buff idle");
  return std::nullopt;
}

L2Perception::ArmorDetector makeArmorDetector(
  const runtime::AutoAimConfig& config)
{
  try {
    auto backend = L2Perception::makeInferenceBackend(config.inference_backend);
    // 模型路径、设备、颜色顺序、归一化以及后端调度参数全部来自 auto_aim.yaml
    // 的 inference 节点，loadAutoAimConfig 已经把 model_path/device 回填进去。
    // 宿主输入恒为 uint8 NHWC BGR；颜色和归一化转换由具体后端完成。
    L2Perception::InferenceModelConfig model_config = config.inference;
    try {
      backend->load(model_config);
    } catch (const std::exception& error) {
      // GPU 插件或驱动（intel-opencl-icd）不在时 compile_model 会抛。退回 CPU 还
      // 能打，空检测器就只能看着，所以 OpenVINO 的非 CPU 设备失败时换 CPU 再试一次。
      const bool retry =
        config.inference_backend == L2Perception::InferenceBackendKind::OpenVino &&
        !model_config.device.starts_with("CPU");
      if (!retry) {
        throw;
      }
      L6Telemetry::logWarn(
        "armor model failed on", model_config.device, error.what(), "; falling back to CPU");
      model_config.device = "CPU";
      backend->load(model_config);
    }

    // Decoder 的字段布局跟着 model_path 走：layout: auto 时按模型输出形状选预设，
    // 再盖上 YAML 写了的阈值。
    L2Perception::ArmorDecoderConfig decoder = config.decoder;
    if (config.auto_layout) {
      decoder = L2Perception::armorDecoderConfigFor(L2Perception::probeOutputSpecs(*backend));
      runtime::applyThresholds(config.decoder_thresholds, decoder);
    }
    const bool yolov8 =
      decoder.contract.tensor_layout == L2Perception::ArmorTensorLayout::FieldsByCandidates;

    L6Telemetry::logInfo(
      "armor model loaded",
      std::string{L2Perception::inferenceBackendName(config.inference_backend)},
      config.model_path.string(), model_config.device,
      "layout", yolov8 ? "yolov8_21" : "yolov5_22", config.auto_layout ? "(auto)" : "(yaml)",
      "output", decoder.contract.output_name, "conf", decoder.confidence_threshold);
    // 预处理保持默认（letterbox 的对齐和填充色对现有模型实测无差别）；传统灯条
    // 精修来自 refiner 节点。
    return L2Perception::ArmorDetector(
      std::move(backend), decoder, L2Perception::ImagePreprocessConfig{},
      config.refiner);
  } catch (const std::exception& error) {
    // 模型或 SDK 不可用时只在启动阶段记录一次；空 Detector 会持续返回安全的空结果。
    L6Telemetry::logError(
      "armor model unavailable",
      std::string{L2Perception::inferenceBackendName(config.inference_backend)},
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
  const std::optional<L3Estimation::TrackedTarget>& target,
  L3Estimation::TrackState track_state,
  const L4Planning::Plan& plan,
  const L5Control::FireDecision& fire,
  bool command_sent,
  const runtime::RuntimeIo& io,
  std::chrono::steady_clock::time_point timestamp,
  int detect_count,
  std::uint64_t tracker_resets,
  std::uint64_t plan_rejects,
  bool pose_ready,
  double pose_wait_ms);
nlohmann::json buffTelemetryFrame(
  const L1Sensor::RobotState& state,
  const L3Estimation::Buff::Addition& addition,
  const std::optional<L3Estimation::Buff::State>& buff_state,
  const L4Planning::Buff::Plan& plan,
  const std::optional<L5Control::SerialCommand>& command,
  std::chrono::steady_clock::time_point timestamp,
  bool command_sent);
/********************************** debug **********************************/

} // namespace

namespace runtime {

AutoAimRuntime::AutoAimRuntime(const std::string &config_path)
    : config_path_(config_path) {}

void AutoAimRuntime::run() {
  running_ = true;
  AutoAimConfig auto_aim_config =
    loadAutoAimConfig("config/auto_aim.yaml");
  auto io = makeRuntimeIo(config_path_);
  if (io->name() == "talos") {
    // Talos 自己提供模式和弹速。硬件调试旁路若覆盖它，small/big_buff 配置会
    // 永远进不了对应链路；信任 fallback 又会把仿真的 projectile.speed 丢掉。
    auto_aim_config.debug.force_work_mode.clear();
    auto_aim_config.plan.impact.trust_fallback_bullet_speed = false;
  }
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    active_io_ = io;
  }
  // 启动时只加载一次模型；每帧仅执行预处理、推理和 Decoder。
  L2Perception::ArmorDetector armor_detector =
    makeArmorDetector(auto_aim_config);

  const bool io_started = io->start();
  if (!io_started) {
    L6Telemetry::logWarn("Failed to start runtime I/O backend", io->name());
  } else {
    L6Telemetry::logInfo("runtime I/O backend started", io->name());
  }
  // L3 Tracker 持有 PnP 和 EKF。标定缺失时 runtime 继续运行检测和显示，
  // 但后续不得生成有效瞄准/开火命令。
  std::optional<L3Estimation::Tracker> tracker;
  const auto& camera_calibration = io->calibration();
  if (!camera_calibration) {
    L6Telemetry::logWarn("Tracker disabled: camera calibration is missing");
  } else {
    tracker.emplace(
      *camera_calibration,
      auto_aim_config.armor,
      auto_aim_config.tracker,
      auto_aim_config.target);
    if (tracker->ready()) {
      L6Telemetry::logInfo("L3 tracker configured");
    } else {
      L6Telemetry::logWarn(
        "Tracker disabled: calibration, armor, or tracker config is invalid");
    }
  }

  // 打符链路完全独立于装甲板链路；只在相机内参和枪管外参齐全时启用，
  // 缺少外参不能用单位阵猜测，否则世界系姿态会整体偏转。
  std::optional<runtime::buff::Runtime> buff_runtime;
  std::optional<runtime::buff::Config> buff_runtime_config;
  if (camera_calibration && camera_calibration->barrelExtrinsicsReady()) {
    try {
      buff_runtime_config = runtime::buff::loadConfig("config/buff.yaml");
      buff_runtime.emplace(*camera_calibration, *buff_runtime_config);
      L6Telemetry::logInfo("Buff runtime configured");
    } catch (const std::exception& error) {
      L6Telemetry::logError("Buff runtime unavailable", error.what());
    }
  } else {
    L6Telemetry::logWarn(
      "Buff runtime disabled: camera calibration or barrel extrinsics are missing");
  }

  L4Planning::Planner planner(auto_aim_config.plan);
  L5Control::Controller controller(
    auto_aim_config.fire,
    auto_aim_config.runtime.command_jump_threshold);

  const auto publishCommand = [&io, io_started](
                                const std::optional<L5Control::SerialCommand>& command) {
    if (io_started && command) {
      io->updateCommand(*command);
    }
  };

  // 规划失败时不能只是“不更新命令”：上一条 shoot=true 在
  // command_timeout 内仍可能被重复发送。这里保留最后角度并立即关火。
  const auto sendAimSafeHold = [&]() {
    publishCommand(controller.safeHold());
  };

  // 会话停止只负责重置算法状态，不直接写串口。否则从装甲板
  // 切到打符时，旧链路的 safe-hold 会与新链路命令在同一帧竞争。
  // 返回值留给调用方做过渡保底，最终每帧最多发布一条命令。
  bool aim_session_active = false;
  const auto stopAimSession = [&]() -> std::optional<L5Control::SerialCommand> {
    if (!aim_session_active) return std::nullopt;
    const auto safe_hold = controller.safeHold();
    if (tracker) {
      tracker->reset();
    }
    planner.reset();
    aim_session_active = false;
    return safe_hold;
  };

  bool buff_session_active = false;
  const auto stopBuffSession = [&]() -> std::optional<L5Control::SerialCommand> {
    if (!buff_runtime || !buff_session_active) return std::nullopt;
    // reset() 会清掉 L5 的最后角度，因此必须先取安全保持命令。
    const auto safe_hold = buff_runtime->safeHold();
    buff_runtime->reset();
    buff_session_active = false;
    return safe_hold;
  };

  /******************************** debug *********************************/
  // 叠加层默认关闭：imshow 的耗时会计进 image_to_plan，而且比赛用的机器
  // 没有显示器，无条件 namedWindow 会直接抛。
  bool overlay_enabled = auto_aim_config.debug.overlay;
  const bool web_enabled = auto_aim_config.debug.web;
  // 画与显分离：Web 也要标注图，但不能 namedWindow/imshow。
  bool draw_debug = overlay_enabled || web_enabled;
  if (overlay_enabled) {
    cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);
  }
  std::uint64_t frame_index = 0;
  // 叠加层要把世界系位姿投回图像，需要一个求解器。这里另建一个与 Tracker
  // 内部同参数的实例，只做重投影、不参与滤波——不为了画图去开 Tracker 的内部。
  std::optional<L3Estimation::PnpSolver> overlay_solver;
  if (draw_debug && camera_calibration) {
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
  // Web 与 PlotJuggler 消费同一份 telemetryFrame；WebSink 只在 web 打开时构造，
  // 写 /dev/shm/newvision_*，由独立的 web.py 进程转成 HTTP。
  std::optional<L6Telemetry::WebSink> web_sink;
  if (web_enabled) {
    web_sink.emplace(
      200, auto_aim_config.debug.web_record_dir,
      auto_aim_config.debug.web_record_max_frames);
    L6Telemetry::logInfo("web debug enabled", "http://0.0.0.0:9000");
    if (!auto_aim_config.debug.web_record_dir.empty()) {
      L6Telemetry::logInfo(
        "web recording to", auto_aim_config.debug.web_record_dir);
    }
  }
  // 在线调参通道：与 WebSink 分开开关。poll 只读 shm，不进瞄准链路。
  L6Telemetry::WebCmd web_cmd;
  auto forced_mode = parseWorkMode(auto_aim_config.debug.force_work_mode);
  const auto applyTuningIfAny = [&]() {
    if (!web_cmd.poll()) {
      return;
    }
    const auto changed = applyWebTuning(auto_aim_config, web_cmd.snapshot());
    if (changed.empty()) {
      return;
    }
    std::string joined;
    for (const auto& key : changed) {
      joined += joined.empty() ? key : ("," + key);
    }
    L6Telemetry::logWarn("web tuning applied", joined);
    const bool forced_mode_changed = std::find(
      changed.begin(), changed.end(), "debug.force_work_mode") != changed.end();
    if (forced_mode_changed) {
      forced_mode = parseWorkMode(auto_aim_config.debug.force_work_mode);
      if (forced_mode) {
        L6Telemetry::logWarn(
          "!!! debug.force_work_mode changed at runtime:",
          L1Sensor::toString(*forced_mode), "- the MCU's WorkMode is being IGNORED");
      } else {
        L6Telemetry::logInfo(
          "debug.force_work_mode cleared at runtime; following MCU WorkMode");
      }
      if (web_sink) {
        web_sink->writeEvent(
          "force_mode_change",
          forced_mode ? nlohmann::json{L1Sensor::toString(*forced_mode)}
                      : nlohmann::json{"mcu"});
      }
    }
    const bool overlay_changed = std::find(
      changed.begin(), changed.end(), "debug.overlay") != changed.end();
    if (overlay_changed &&
        overlay_enabled != auto_aim_config.debug.overlay) {
      try {
        if (auto_aim_config.debug.overlay) {
          cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);
        } else {
          cv::destroyWindow("auto_aim");
        }
        overlay_enabled = auto_aim_config.debug.overlay;
        draw_debug = overlay_enabled || web_enabled;
        if (draw_debug && !overlay_solver && camera_calibration) {
          overlay_solver.emplace(*camera_calibration, auto_aim_config.armor);
        }
      } catch (const cv::Exception& error) {
        auto_aim_config.debug.overlay = overlay_enabled;
        L6Telemetry::logWarn("web overlay toggle failed", error.what());
      }
    }
    // 热更进活对象。Tracker 持有当前 EKF，必须连 target 一起改，
    // 否则要等丢目标重建才生效，调参时看不出差别。
    if (tracker) {
      tracker->setTargetConfig(auto_aim_config.target);
    }
    planner.setImpactConfig(auto_aim_config.plan.impact);
    controller.setFireConfig(auto_aim_config.fire);
    if (auto_aim_config.debug.camera_exposure_ms && auto_aim_config.debug.camera_gain) {
      const bool ok = io->setExposureGain(
        *auto_aim_config.debug.camera_exposure_ms, *auto_aim_config.debug.camera_gain);
      if (!ok) {
        L6Telemetry::logWarn(
          "camera exposure/gain not applied (driver does not support hot update)");
      }
    }
    if (web_sink) {
      web_sink->writeEvent("tuning_applied", joined);
    }
  };
  /******************************** debug *********************************/

  // 调试旁路：强制 WorkMode。Web 热更会同步刷新 forced_mode。
  if (forced_mode) {
    L6Telemetry::logWarn(
      "!!! debug.force_work_mode is ACTIVE:", L1Sensor::toString(*forced_mode),
      "- the MCU's WorkMode is being IGNORED. Clear this key before a match.");
  }
  if (auto_aim_config.plan.impact.trust_fallback_bullet_speed) {
    L6Telemetry::logWarn(
      "!!! planning.trust_fallback_bullet_speed is ACTIVE: ballistics assume",
      auto_aim_config.plan.impact.fallback_bullet_speed,
      "m/s regardless of what the MCU reports. Set it back to false once the"
      " MCU sends a real bullet speed.");
  }

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;
  // "规划结束 -> 串口发出"的实测耗时。本帧的值要等规划做完才知道，所以
  // 用上一帧的量代入本帧的延迟链；这一段帧间基本恒定。
  double measured_plan_to_send = 0.0;
  // L4 拒绝出计划的累计帧数。拒绝时 L5 会原样重发上一条命令（safeHold），
  // 云台角就此冻结一帧，所以这个数直接对应"命令被冻住了几帧"。
  std::uint64_t plan_reject_count = 0;
  TalosMetrics talos_metrics;
  // 事件边沿：只在变化时 writeEvent，避免每帧刷日志树。
  std::optional<L1Sensor::WorkMode> last_mode_for_event;
  std::uint64_t last_reset_for_event = 0;
  int last_fire_reason_for_event = -1;

  // CAM 是驱动线程实际采到的帧率；ALG 是主循环成功取帧的吞吐率。
  // 两者分开后，一眼就能判断瓶颈在相机，还是算法来不及消费。
  L6Telemetry::FpsCounter algorithm_fps;
  auto camera_sample_at = std::chrono::steady_clock::now();
  auto last_camera_frames = io->stats().camera_captured_frames;
  double camera_fps = 0.0;
  const auto updatePerformanceCounters = [&]() {
    const auto now = std::chrono::steady_clock::now();
    algorithm_fps.update(now);
    const double elapsed =
      std::chrono::duration<double>(now - camera_sample_at).count();
    if (elapsed < 1.0) {
      return;
    }
    const auto current = io->stats().camera_captured_frames;
    const auto frames = current >= last_camera_frames
      ? current - last_camera_frames : 0;
    camera_fps = static_cast<double>(frames) / elapsed;
    last_camera_frames = current;
    camera_sample_at = now;
  };
  const auto appendPerformanceTelemetry = [&](nlohmann::json& data) {
    const auto runtime_stats = io->stats();
    data["runtime"]["camera_fps"] = camera_fps;
    data["runtime"]["algorithm_fps"] = algorithm_fps.fps();
    data["runtime"]["camera_captured"] = runtime_stats.camera_captured_frames;
    data["runtime"]["camera_dropped"] = runtime_stats.camera_dropped_frames;
    data["runtime"]["algorithm_received"] = runtime_stats.algorithm_received_frames;
    if (web_sink) {
      const auto web_stats = web_sink->stats();
      data["runtime"]["web_fps"] = web_stats.encode_fps;
      data["runtime"]["web_submitted"] = web_stats.submitted_frames;
      data["runtime"]["web_encoded"] = web_stats.encoded_frames;
      data["runtime"]["web_dropped"] = web_stats.dropped_frames;
      data["runtime"]["web_encode_ms"] = web_stats.encode_ms;
    }
  };

  // 单线程同步是设计选择不是待办：自瞄的代价是开火那一刻的位置误差而不是
  // 帧率，异步流水线换来吞吐、代价是结果多滞后一帧，那一帧会进
  // Delay::image_to_plan 再被 v_yaw 放大成瞄准偏差。真正降低单帧延迟的并行
  // 在推理内部（num_threads / scheduling_core_type），那层已经开着。
  // tools/LatestBuffer 是给将来的取图线程预留的，当前管线不用。
  while (running_) {
    //获取相机帧和时间辍
    if (!io->read(frame, timestamp)) {
      if (!running_) {
        break;
      }
      // 读帧超时期间没有新观测，不继续续期上一帧的开火位。
      if (buff_session_active) {
        publishCommand(stopBuffSession());
      } else {
        sendAimSafeHold();
      }
      continue;
    }
    updatePerformanceCounters();
    bool wrote_web_series = false;
    const auto state = io->latestState();
    if (!io_started || !state) {
      const auto aim_hold = stopAimSession();
      const auto buff_hold = stopBuffSession();
      publishCommand(buff_hold ? buff_hold : aim_hold);
    } else {
      // 覆盖只改分派用的模式，state 本身不动——遥测和日志仍然反映下位机
      // 真正上报的值，否则排查时会看不出电控到底给没给对模式。
      const L1Sensor::WorkMode mode = forced_mode.value_or(state->mode);
      switch (mode) {
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost: {
          const auto transition_hold = stopBuffSession();
          aim_session_active = true;
          // L2: 检测并保留敌方装甲板。
          auto armors = armor_detector.detect(frame);
          std::erase_if(armors, [&state](const auto& armor) {
            return !isEnemyArmor(armor.color, state->enemy_color);
          });

          // 曝光时刻的姿态要晚 serial pose_delay_ms 才到，图像刚到手时还没收到。
          // 放在检测之后等，等待与检测重叠，不白白加延迟；等不到（串口断流）
          // 就用边界值，遥测里 pose_ready=0 会标出来。
          const auto wait_start = std::chrono::steady_clock::now();
          const bool pose_ready = io->waitPose(timestamp);
          const double pose_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
          const auto image_pose = io->gimbalPoseAt(timestamp);

          // L3: PnP、状态机与整车 EKF。
          std::optional<L3Estimation::TrackedTarget> target;
          if (tracker && tracker->ready()) {
            target = tracker->track(armors, image_pose, timestamp);
          }
          const auto track_state = tracker
            ? tracker->state()
            : L3Estimation::TrackState::Lost;

          if (tracker) {
            if (const auto truth = io->groundTruth()) {
              updateTalosMetrics(
                talos_metrics, tracker->observations(), *truth, state->enemy_color);
            }
          }

          // 规划与开火判定使用推理结束时刻。
          const auto plan_time = std::chrono::steady_clock::now();
          // 这里要的是“现在”的云台角，不是过去某一时刻的插值。“现在”之后不可能
          // 有采样，走 gimbalPoseAt(now()) 只会每帧都撞越界分支、白白计一次数。
          const auto actual_pose = io->latestGimbalPose();

          // L4: 预测命中时刻、选板并解算弹道。
          L4Planning::PlanInput plan_input;
          plan_input.target = target;
          plan_input.robot_state = *state;
          plan_input.plan_time = plan_time;
          plan_input.to_now = true;
          plan_input.plan_to_send = measured_plan_to_send;
          const auto plan = planner.plan(plan_input);
          if (!plan.valid()) {
            ++plan_reject_count;
          }

          // L5: 开火判定、命令跳变检查和安全保持。
          const auto command = controller.update(
            target,
            track_state,
            plan,
            actual_pose);

          // 从打符切回装甲板的首帧若还没有有效规划，保持打符
          // 最后角度并关火，避免回跳到上一次装甲板会话的陈旧角度。
          const auto command_to_publish =
            transition_hold && !plan.valid() ? transition_hold : command;

          // L1: 下发 L5 生成的控制命令，并量出本帧规划到发送的耗时，
          // 供下一帧的延迟链使用。
          publishCommand(command_to_publish);

          // 规划到发送的实测耗时必须在 updateCommand 之后**立刻**取。
          // 放到叠加层之后的话，画图的几毫秒会被算进 plan_to_send，而恰恰
          // 只有开着叠加层调试时才会去看这个数。
          measured_plan_to_send = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - plan_time).count();

          /*************************** debug ****************************/
          // 全部排在命令下发和延迟测量之后，不占用瞄准链路的时间预算。
          if (plotter || web_sink) {
            // 三目里直接放 observations() 会按值合成公共类型，等于每帧拷一份
            // 整个 vector；用一个空的静态量接住 tracker 为空的分支。
            static const std::vector<L3Estimation::Armor> kNoObservations;
            const auto& observations =
              tracker ? tracker->observations() : kNoObservations;
            auto data = telemetryFrame(
              image_pose, *state, observations, target, track_state, plan,
              controller.lastDecision(), command_to_publish.has_value(), *io,
              timestamp, tracker ? tracker->detectCount() : 0,
              tracker ? tracker->resetCount() : 0, plan_reject_count,
              pose_ready, pose_wait_ms);
            appendPerformanceTelemetry(data);
            if (plotter) {
              (void)plotter->send(data);
            }
            if (web_sink) {
              web_sink->writeSeries(data);
              wrote_web_series = true;
            }
          }
          // 画与显分离：imshow 只在 overlay 开启时才做；Web 要的只是标注帧。
          // overlay_every 与 web_frame_every 独立，同一帧最多画一次。
          const bool draw_this_frame =
            overlay_solver && tracker &&
            (frame_index % auto_aim_config.debug.overlay_every == 0 ||
             (web_sink && frame_index % auto_aim_config.debug.web_frame_every == 0));
          if (draw_this_frame) {
            overlay_solver->set_R_world_barrel(image_pose);
            L6Telemetry::drawAimOverlay(
              frame,
              {.detections = armors,
               .observations = tracker->observations(),
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
          const auto transition_hold = stopAimSession();
          if (!buff_runtime) {
            publishCommand(transition_hold);
            break;
          }
          buff_session_active = true;

          auto detections = buff_runtime->detect(frame, state->enemy_color);
          const auto wait_start = std::chrono::steady_clock::now();
          const bool pose_ready = io->waitPose(timestamp);
          (void)pose_ready;
          (void)wait_start;
          const auto image_pose = io->gimbalPoseAt(timestamp);
          if (!image_pose) {
            publishCommand(transition_hold ? transition_hold : buff_runtime->safeHold());
            break;
          }

          const auto camera_pose = L1Sensor::Buff::composeCameraPose(
            *image_pose, *camera_calibration);
          if (!camera_pose) {
            publishCommand(transition_hold ? transition_hold : buff_runtime->safeHold());
            break;
          }
          const auto plan_time = std::chrono::steady_clock::now();
          const auto result = buff_runtime->update(
            std::move(detections), camera_pose->q_world_camera,
            camera_pose->t_world_camera,
            timestamp, plan_time, state->bullet_speed);
          // 新打符会话首帧尚未形成有效规划时，优先保持装甲板链路
          // 的最后角度并关火；有效计划则只发布打符命令。
          const auto command_to_publish =
            transition_hold && !result.plan.valid ? transition_hold : result.command;
          publishCommand(command_to_publish);
          if (web_sink) {
            auto data = buffTelemetryFrame(
              *state, result.addition, result.state, result.plan,
              command_to_publish, timestamp, command_to_publish.has_value());
            appendPerformanceTelemetry(data);
            web_sink->writeSeries(data);
            wrote_web_series = true;
          }
          if (buff_runtime_config &&
              (buff_runtime_config->overlay ||
               (web_sink && frame_index % auto_aim_config.debug.web_frame_every == 0))) {
            L6Telemetry::Buff::drawOverlay(
              frame, result.detections, result.addition, result.plan);
          }
          break;
        }

        case L1Sensor::WorkMode::Idle:
        default:
          {
            const auto aim_hold = stopAimSession();
            const auto buff_hold = stopBuffSession();
            publishCommand(buff_hold ? buff_hold : aim_hold);
          }
          break;
      }
    }

    /******************************* debug ********************************/
    // Idle、相机状态未就绪等分支也持续发布性能指标，避免网页上的帧率冻结。
    if (web_sink && !wrote_web_series) {
      nlohmann::json data;
      data["t"] = std::chrono::duration<double>(timestamp.time_since_epoch()).count();
      appendPerformanceTelemetry(data);
      web_sink->writeSeries(data);
    }
    // 在线调参：每 10 帧读一次 shm。文件不存在时 poll 立即返回，代价很小。
    if (frame_index % 10 == 0) {
      applyTuningIfAny();
    }
    // 关键事件进 Log 树：模式切换、EKF 重建、开火拒绝原因变化。
    if (web_sink) {
      if (state) {
        if (last_mode_for_event != state->mode) {
          web_sink->writeEvent(
            "mode_change", std::string{L1Sensor::toString(state->mode)});
          last_mode_for_event = state->mode;
        }
      }
      if (tracker) {
        const auto resets = tracker->resetCount();
        if (resets != last_reset_for_event) {
          web_sink->writeEvent("tracker_reset", static_cast<double>(resets));
          last_reset_for_event = resets;
        }
      }
      const int reason = controller.lastDecision().reasons.empty()
        ? -1
        : static_cast<int>(controller.lastDecision().reasons.front());
      if (reason != last_fire_reason_for_event && reason >= 0) {
        web_sink->writeEvent("fire_reject", reason);
        last_fire_reason_for_event = reason;
      }
    }
    // Web 写帧与 imshow 分开：writeFrame 只复制最新帧，缩放和 JPEG 压缩由
    // 后台单槽线程完成，不会因为浏览器或无线网络变慢而反压视觉主循环。
    // 与绘制共用同一个 frame_index：++ 在后面，保证「画的帧 == 写的帧」。
    if (web_sink && frame_index % auto_aim_config.debug.web_frame_every == 0) {
      web_sink->writeFrame(frame);
    }
    ++frame_index;
    if (overlay_enabled) {
      cv::imshow("auto_aim", frame);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        running_ = false;
      }
    }
    /******************************* debug ********************************/
  }

  const auto aim_hold = stopAimSession();
  const auto buff_hold = stopBuffSession();
  publishCommand(buff_hold ? buff_hold : aim_hold);
  io->stop();
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (active_io_ == io) {
      active_io_.reset();
    }
  }
  if (overlay_enabled) {
    cv::destroyWindow("auto_aim");
  }
}

void AutoAimRuntime::stop() {
  running_ = false;
  std::shared_ptr<RuntimeIo> io;
  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    io = active_io_;
  }
  if (io) {
    io->stop();
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
  const std::optional<L3Estimation::TrackedTarget>& target,
  L3Estimation::TrackState track_state,
  const L4Planning::Plan& plan,
  const L5Control::FireDecision& fire,
  bool command_sent,
  const runtime::RuntimeIo& io,
  std::chrono::steady_clock::time_point timestamp,
  int detect_count,
  std::uint64_t tracker_resets,
  std::uint64_t plan_rejects,
  bool pose_ready,
  double pose_wait_ms)
{
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  nlohmann::json data;

  // 帧曝光时刻，单位 s。**必须发**：不发的话 PlotJuggler 只能按 UDP 到达时刻
  // 排点，而本循环的周期本身就抖（实测帧间 10~28 ms），平滑的斜坡会被画成
  // 忽快忽慢的折线，看起来像台阶——那是坐标轴的假象，不是信号的。这次现场
  // 就先被它误导过一轮。在 UDP/JSON 插件里把它选成 timestamp 字段。
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

  // serial: 串口健康度。丢包和姿态越界都只累加计数、不打日志，只能从这里看。
  // pose_after 高说明图像时间戳比最新 IMU 采样还新——曝光中点补偿之外，读出
  // 与 USB 传输耗时仍未补；pose_before 高则是图像太老或姿态历史太短，成因相反。
  // rx_skipped_bytes 恒为 0 说明下位机只是空转 seq，跟着 rx_dropped 一起涨才
  // 说明有第三种 SOF 的帧被静默吃掉了。
  // pose_wait 是检测做完后还要等姿态的时间；pose_ready=0 说明等满了也没等到。
  data["serial"]["pose_ready"] = pose_ready ? 1 : 0;
  data["serial"]["pose_wait"] = pose_wait_ms;
  const runtime::RuntimeIoStats io_stats = io.stats();
  data["serial"]["rx"] = io_stats.received_states;
  data["serial"]["rx_dropped"] = io_stats.dropped_packets;
  data["serial"]["rx_skipped_bytes"] = io_stats.skipped_bytes;
  data["serial"]["tx"] = io_stats.sent_commands;
  data["serial"]["tx_failed"] = io_stats.failed_commands;
  data["serial"]["pose_before_history"] = io_stats.pose_before_history;
  data["serial"]["pose_after_history"] = io_stats.pose_after_history;

  // track: 状态机与本帧真正进滤波器的观测数量。
  data["track"]["state"] = static_cast<int>(track_state);
  data["track"]["n_obs"] = static_cast<int>(observations.size());
  // 瞬时枚举看不出"每隔几帧重建一次 EKF"——那在状态图上只是一个单帧尖峰，
  // 跟采样率一撞就完全看不见。这三个是单调计数器：只要发生过就一定留下台阶。
  // aim/rejects 是 L4 拒绝出计划的累计帧数，拒绝时 L5 原样重发上一条命令、
  // 云台角冻结一帧，所以它直接对应"命令被冻住了几帧"。
  data["track"]["detect_count"] = detect_count;
  data["track"]["resets"] = tracker_resets;
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

  // ekf: 整车状态全 13 维，加一致性统计。下标顺序见 TrackedTarget::kStateSize
  // 的注释，前十一维不可改。
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
    data["ekf"]["dr"] = x[9];
    data["ekf"]["dz"] = x[10];
    data["ekf"]["dz1"] = x[11];
    data["ekf"]["dz2"] = x[12];
    data["ekf"]["last_id"] = target->last_id;
    data["ekf"]["jumped"] = target->jumped ? 1 : 0;

    // 残差按分量发。只发一个 NIS 标量的话，超标时无法定位是哪一维在超。
    const auto& ekf_data = target->ekf().data;
    data["ekf"]["res_yaw"] = ekf_data.at("residual_yaw");
    data["ekf"]["res_pitch"] = ekf_data.at("residual_pitch");
    data["ekf"]["res_distance"] = ekf_data.at("residual_distance");
    data["ekf"]["res_angle"] = ekf_data.at("residual_angle");
    data["ekf"]["nis"] = ekf_data.at("nis");
    data["ekf"]["nees"] = ekf_data.at("nees");
    data["ekf"]["nis_fail"] = ekf_data.at("nis_fail");
    data["ekf"]["nis_fail_rate"] = ekf_data.at("recent_nis_failures");
  }

  // aim: L4 规划出的云台目标姿态。**与 gimbal/ 分开**，两者同图即跟随误差。
  data["aim"]["status"] = static_cast<int>(plan.status);
  data["aim"]["reason"] = static_cast<int>(plan.reason);
  if (plan.valid()) {
    data["aim"]["yaw"] = plan.aim.yaw * kRadToDeg;
    data["aim"]["pitch"] = plan.aim.pitch * kRadToDeg;
    // 射击轨迹原值。**必须和 aim/yaw 画在同一张图上**：过渡段调的就是这
    // 两条线怎么分开又怎么合上，只画一条完全看不出过渡做没做、做过头没有。
    // 跟随段两者重合是正常的，不是数据重复。
    data["aim"]["shoot_yaw"] = plan.aim.shoot_yaw * kRadToDeg;
    data["aim"]["shoot_pitch"] = plan.aim.shoot_pitch * kRadToDeg;
  }
  data["aim"]["blending"] = plan.aim.blending ? 1 : 0;
  // 过渡段规划出的峰值角加速度，对着 planning.blend 里配的上限看。贴着上限
  // 走是正常的；acc_limited 置 1 才是问题——那说明拉到最长时长仍然超限，
  // 重合度上不去是云台能力的物理限制，不是参数没调好。
  data["aim"]["blend_peak_acc"] = plan.blend.peak_yaw_acceleration;
  data["aim"]["blend_acc_limited"] = plan.blend.acceleration_limited ? 1 : 0;
  // 过渡终点比切板时刻晚了多少毫秒。稳定在一个图像帧周期附近是正常的量化
  // 误差；明显更大说明 blend.horizon_ms 不够长，切板发现得太晚。
  data["aim"]["blend_late_ms"] = plan.blend.late * 1e3;
  data["aim"]["armor_id"] = plan.fire ? plan.fire->armor_id : -1;

  // delay: 五段延迟链。绝不合并成一个标量——上车标定 send_to_control 时
  // 要能看出是哪一段在变。
  const L4Planning::Delay& delay = plan.timing.delay;
  data["delay"]["image_to_plan"] = delay.image_to_plan * 1e3;
  data["delay"]["plan_to_send"] = delay.plan_to_send * 1e3;
  data["delay"]["send_to_control"] = delay.send_to_control * 1e3;
  data["delay"]["control_to_fire"] = delay.control_to_fire * 1e3;
  data["delay"]["fire_to_hit"] = delay.fire_to_hit * 1e3;
  data["delay"]["before_fire"] = delay.beforeFire() * 1e3;

  // fire: 可行性与实际下发分开。feasible=1 而 shoot=0 就是被 shoot_enable
  // 或跳变检查拦下来了，reason 给出第一条原因（-1 表示无拒绝）。
  data["fire"]["feasible"] = fire.fire_feasible ? 1 : 0;
  data["fire"]["shoot"] = fire.shoot ? 1 : 0;
  data["fire"]["sent"] = command_sent ? 1 : 0;
  data["fire"]["yaw_err"] = fire.yaw_error * kRadToDeg;
  data["fire"]["pitch_err"] = fire.pitch_error * kRadToDeg;
  data["fire"]["tol_yaw"] = fire.tolerance.yaw * kRadToDeg;
  data["fire"]["tol_pitch"] = fire.tolerance.pitch * kRadToDeg;
  data["fire"]["reason"] =
    fire.reasons.empty() ? -1 : static_cast<int>(fire.reasons.front());

  return data;
}

// 打符遥测。与 armor 的 telemetryFrame 同一套约定：分支隔离、degree/ms。
// buff/ 下是能量机关自己的状态，不与 armor 的 ekf/aim 混层。
nlohmann::json buffTelemetryFrame(
  const L1Sensor::RobotState& state,
  const L3Estimation::Buff::Addition& addition,
  const std::optional<L3Estimation::Buff::State>& buff_state,
  const L4Planning::Buff::Plan& plan,
  const std::optional<L5Control::SerialCommand>& command,
  std::chrono::steady_clock::time_point timestamp,
  bool command_sent)
{
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  nlohmann::json data;

  data["t"] = std::chrono::duration<double>(timestamp.time_since_epoch()).count();

  data["gimbal"]["bullet_speed"] = state.bullet_speed;

  data["buff"]["n_icon"] = static_cast<int>(addition.tracked.size());
  data["buff"]["n_pred"] = static_cast<int>(addition.predicted.size());

  if (buff_state) {
    data["buff"]["x"] = buff_state->x;
    data["buff"]["y"] = buff_state->y;
    data["buff"]["z"] = buff_state->z;
    data["buff"]["rot_speed"] = buff_state->rotation_speed;
    data["buff"]["rot_angle"] = buff_state->rotation_angle * kRadToDeg;
    data["buff"]["face_yaw"] = buff_state->face_yaw * kRadToDeg;
    data["buff"]["update_count"] = static_cast<double>(buff_state->update_count);
    data["buff"]["sine_valid"] = buff_state->sine_valid ? 1 : 0;
    data["buff"]["sine_C"] = buff_state->sine_C;
    data["buff"]["sine_v"] = buff_state->sine_v;
    data["buff"]["sine_a"] = buff_state->sine_a;
    data["buff"]["sine_omega"] = buff_state->sine_omega;
    data["buff"]["sine_phase"] = buff_state->sine_phase * kRadToDeg;
    data["buff"]["use_pred_speed"] = buff_state->use_prediction_speed ? 1 : 0;
    data["buff"]["pred_cost"] = buff_state->prediction_cost;
    int inactive = 0;
    for (bool flag : buff_state->inactive) {
      inactive += flag ? 1 : 0;
    }
    data["buff"]["inactive"] = inactive;
  }

  data["buff"]["plan_valid"] = plan.valid ? 1 : 0;
  data["buff"]["degraded"] = plan.degraded ? 1 : 0;
  data["buff"]["pre_aim"] = plan.pre_aim ? 1 : 0;
  data["buff"]["shoot"] = plan.algorithm_shoot ? 1 : 0;
  data["buff"]["blade"] = plan.blade_index;
  data["buff"]["fly_time"] = plan.fly_time * 1e3;
  data["buff"]["bullet_trusted"] = plan.bullet_speed.trusted ? 1 : 0;
  if (plan.valid) {
    data["aim"]["yaw"] = plan.yaw * kRadToDeg;
    data["aim"]["pitch"] = plan.pitch * kRadToDeg;
  }
  if (command) {
    data["aim"]["cmd_yaw"] = command->yaw * kRadToDeg;
    data["aim"]["cmd_pitch"] = command->pitch * kRadToDeg;
    data["fire"]["shoot"] = command->shoot ? 1 : 0;
    data["fire"]["sent"] = command_sent ? 1 : 0;
  }

  return data;
}

}  // namespace
/********************************** debug **********************************/
