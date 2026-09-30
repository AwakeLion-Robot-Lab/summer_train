#include "runtime/auto_aim_runtime.hpp"
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l4_planning/planner.hpp"
#include "l4_planning/planner_config.hpp"
#include "l5_control/controller.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "l6_telemetry/udp_json_sender.hpp"
#include "runtime/auto_aim_config.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/l4_target_adapter.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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

std::optional<L1Sensor::WorkMode> parseWorkMode(const std::string& name)
{
  if (name.empty()) return std::nullopt;
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
  const AutoAimConfig auto_aim_config =
    loadAutoAimConfig("config/auto_aim.yaml");
  L4Planning::PlannerTuning planner_tuning;
  try {
    planner_tuning = L4Planning::loadPlannerTuning(
      "config/planner_config.yaml");
  } catch (const std::exception& error) {
    L6Telemetry::logError(
      "failed to load planner configuration", error.what());
    running_ = false;
    return;
  }
  auto camera = std::make_shared<L1Sensor::Camera>(config_path_);
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    active_camera_ = camera;
  }
  // 启动时只加载一次模型；每帧仅执行预处理、推理和 Decoder。
  L2Perception::ArmorDetector armor_detector =
    makeArmorDetector(auto_aim_config);

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
  } else {
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

  // 规划失败时不能只是“不更新命令”：上一条 shoot=true 在
  // command_timeout 内仍可能被重复发送。这里保留最后角度并立即关火。
  const auto sendSafeHold = [&]() {
    if (serial_started) {
      if (const auto command = controller.safeHold()) {
        serial.updateCommand(*command);
      }
    }
  };

  const auto stopAimSession = [&]() {
    if (tracker) {
      tracker->reset();
    }
    planner.resetTracking();
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
  const auto forced_mode = parseWorkMode(auto_aim_config.debug.force_work_mode);
  if (forced_mode) {
    L6Telemetry::logWarn(
      "!!! debug.force_work_mode is ACTIVE:", L1Sensor::toString(*forced_mode),
      "- the MCU's WorkMode is being IGNORED. Clear this key before a match.");
  }
  std::uint64_t plan_reject_count = 0;
  bool using_default_bullet_speed = false;
  /******************************** debug *********************************/

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;
  // "规划结束 -> 串口发出"的实测耗时。本帧的值要等规划做完才知道，所以
  // 用上一帧的量代入本帧的延迟链；这一段帧间基本恒定。
  // 单线程同步是设计选择不是待办：自瞄的代价是开火那一刻的位置误差而不是
  // 帧率，异步流水线换来吞吐、代价是结果多滞后一帧，那一帧会进
  // Delay::image_to_plan 再被 v_yaw 放大成瞄准偏差。真正降低单帧延迟的并行
  // 在推理内部（num_threads / scheduling_core_type），那层已经开着。
  // tools/LatestBuffer 是给将来的取图线程预留的，当前管线不用。
  while (running_) {
    //获取相机帧和时间辍
    if (!camera->read(frame, timestamp)) {
      if (!running_) {
        sendSafeHold();
        break;
      }
      // 读帧超时期间没有新观测，不继续续期上一帧的开火位。
      sendSafeHold();
      continue;
    }
    const auto state = serial.latestState();
    if (!serial_started || !state) {
      stopAimSession();
    } else {
      const L1Sensor::WorkMode mode = forced_mode.value_or(state->mode);
      switch (mode) {
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost:
        case L1Sensor::WorkMode::Idle: {
          const bool aiming = mode != L1Sensor::WorkMode::Idle;
          // 等图像之后的姿态包到齐，再用前后两包插值，避免转动中把最新姿态
          // 当作曝光时刻姿态。
          if (!serial.waitPose(timestamp)) {
            L6Telemetry::logDebug("gimbal pose wait timeout");
          }
          const auto image_pose = serial.gimbalPoseAt(timestamp);

          // L2: 用上一帧整车状态生成网络 ROI 和侧灯条 ROI；Lost 时网络 ROI
          // 自动退化为整图。侧灯条作为额外端点观测交给 IESKF。
          std::optional<cv::Rect> light_roi;
          std::optional<cv::Rect> net_roi;
          std::vector<L2Perception::LightHint> light_hints;
          if (tracker && tracker->ready()) {
            light_roi = tracker->lightRoi(image_pose, timestamp, frame.size());
            light_hints = tracker->lightHints(image_pose, timestamp);
            net_roi = tracker->netFocusRoi(
              image_pose, timestamp, frame.size(),
              armor_detector.net_aspect_ratio());
          }
          const auto detect = [&](const std::optional<cv::Rect>& roi) {
            auto result = armor_detector.detectFrame(
              frame, light_roi, roi,
              enemyArmorColor(state->enemy_color),
              light_hints);
            std::erase_if(result.armors, [&state](const auto& armor) {
              return !isEnemyArmor(armor.color, state->enemy_color);
            });
            return result;
          };
          auto perception = detect(net_roi);
          if (tracker && tracker->ready()) {
            if (const auto init_roi = tracker->initRoi(
                  perception.armors, image_pose, timestamp, frame.size(),
                  armor_detector.net_aspect_ratio())) {
              auto refined = detect(*init_roi);
              if (!refined.armors.empty()) {
                perception = std::move(refined);
              }
            }
          }

          // L3: 完整板与独立灯条联合关联，更新整车 IESKF。
          std::optional<L3Estimation::EskfTarget> target;
          if (tracker && tracker->ready()) {
            target = tracker->track(
              perception.armors, perception.lights, image_pose, timestamp);
          }
          const auto track_state = tracker
            ? tracker->state()
            : L3Estimation::TrackState::Lost;

          L4Planning::AimPlan plan;
          std::optional<L5Control::SerialCommand> command;
          if (aiming) {
            // L4/L5 保留 Bruce0178 分支现有实现：动态窗口选板、弹道迭代与
            // 开火闸门均不被目标分支覆盖。
            const auto plan_time = std::chrono::steady_clock::now();
            const auto actual_pose = serial.gimbalPoseAt(plan_time);
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
              serial.updateCommand(*command);
            }
          } else {
            // Idle 继续预热识别与预测，但不运行选板、不输出开火命令。
            planner.resetTracking();
            sendSafeHold();
          }

          // 规划到发送的实测耗时必须在 updateCommand 之后**立刻**取。
          // 放到叠加层之后的话，画图的几毫秒会被算进 plan_to_send，而恰恰
          // 只有开着叠加层调试时才会去看这个数。
          /*************************** debug ****************************/
          // 全部排在命令下发和延迟测量之后，不占用瞄准链路的时间预算。
          if (plotter) {
            static const std::vector<L3Estimation::Armor> kNoObservations;
            (void)plotter->send(telemetryFrame(
              image_pose, *state, kNoObservations, target, track_state, plan,
              controller.lastDecision(), command.has_value(), serial, timestamp,
              tracker ? tracker->lastMatchCount() : 0,
              tracker ? tracker->dropCount() : 0, plan_reject_count));
          }
          if (overlay_solver && tracker &&
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
          // 小能量机后续从这个独立入口接入，不复用装甲板 Tracker。
          stopAimSession();
          break;

        case L1Sensor::WorkMode::BigBuff:
          // 大能量机保留独立模型、跟踪和预测参数入口。
          stopAimSession();
          break;

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
      if (key == 27 || key == 'q' || key == 'Q') {
        running_ = false;
      }
    }
    /******************************* debug ********************************/
  }

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
