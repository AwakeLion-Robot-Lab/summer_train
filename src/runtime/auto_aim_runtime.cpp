#include "runtime/auto_aim_runtime.hpp"
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/armor/tracker.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l5_control/controller.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "l6_telemetry/udp_json_sender.hpp"
#include "runtime/auto_aim_config.hpp"
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
    const L2Perception::InferenceModelConfig& model_config = config.inference;
    backend->load(model_config);

    L6Telemetry::logInfo(
      "armor model loaded",
      std::string{L2Perception::inferenceBackendName(config.inference_backend)},
      config.model_path.string(), model_config.device);
    // Decoder 的字段布局跟着 model_path 走；预处理保持默认（letterbox 的对齐和
    // 填充色对现有模型实测无差别）；传统灯条精修来自 refiner 节点。
    return L2Perception::ArmorDetector(
      std::move(backend), config.decoder, L2Perception::ImagePreprocessConfig{},
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
  // L3 Tracker 持有 PnP 和 EKF。标定缺失时 runtime 继续运行检测和显示，
  // 但后续不得生成有效瞄准/开火命令。
  std::optional<L3Estimation::Tracker> tracker;
  const auto& camera_calibration = camera->calibration();
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

  L4Planning::Planner planner(auto_aim_config.plan);
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
    planner.reset();
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
  /******************************** debug *********************************/

  // 调试旁路：强制 WorkMode。启动时解析一次，循环里只做覆盖。
  const auto forced_mode = parseWorkMode(auto_aim_config.debug.force_work_mode);
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
      // 覆盖只改分派用的模式，state 本身不动——遥测和日志仍然反映下位机
      // 真正上报的值，否则排查时会看不出电控到底给没给对模式。
      const L1Sensor::WorkMode mode = forced_mode.value_or(state->mode);
      switch (mode) {
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost: {
          const auto image_pose = serial.gimbalPoseAt(timestamp);

          // L2: 检测并保留敌方装甲板。
          auto armors = armor_detector.detect(frame);
          std::erase_if(armors, [&state](const auto& armor) {
            return !isEnemyArmor(armor.color, state->enemy_color);
          });

          // L3: PnP、状态机与整车 EKF。
          std::optional<L3Estimation::TrackedTarget> target;
          if (tracker && tracker->ready()) {
            target = tracker->track(armors, image_pose, timestamp);
          }
          const auto track_state = tracker
            ? tracker->state()
            : L3Estimation::TrackState::Lost;

          // 规划与开火判定使用推理结束时刻。
          const auto plan_time = std::chrono::steady_clock::now();
          // 这里要的是“现在”的云台角，不是过去某一时刻的插值。“现在”之后不可能
          // 有采样，走 gimbalPoseAt(now()) 只会每帧都撞越界分支、白白计一次数。
          const auto actual_pose = serial.latestGimbalPose();

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

          // L1: 下发 L5 生成的控制命令，并量出本帧规划到发送的耗时，
          // 供下一帧的延迟链使用。
          if (command) {
            serial.updateCommand(*command);
          }

          // 规划到发送的实测耗时必须在 updateCommand 之后**立刻**取。
          // 放到叠加层之后的话，画图的几毫秒会被算进 plan_to_send，而恰恰
          // 只有开着叠加层调试时才会去看这个数。
          measured_plan_to_send = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - plan_time).count();

          /*************************** debug ****************************/
          // 全部排在命令下发和延迟测量之后，不占用瞄准链路的时间预算。
          if (plotter) {
            // 三目里直接放 observations() 会按值合成公共类型，等于每帧拷一份
            // 整个 vector；用一个空的静态量接住 tracker 为空的分支。
            static const std::vector<L3Estimation::Armor> kNoObservations;
            const auto& observations =
              tracker ? tracker->observations() : kNoObservations;
            (void)plotter->send(telemetryFrame(
              image_pose, *state, observations, target, track_state, plan,
              controller.lastDecision(), command.has_value(), serial,
              timestamp, tracker ? tracker->detectCount() : 0,
              tracker ? tracker->resetCount() : 0, plan_reject_count));
          }
          if (overlay_solver && tracker &&
              frame_index % auto_aim_config.debug.overlay_every == 0) {
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
          // 小能量机后续从这个独立入口接入，不复用装甲板 Tracker。
          stopAimSession();
          break;

        case L1Sensor::WorkMode::BigBuff:
          // 大能量机保留独立模型、跟踪和预测参数入口。
          stopAimSession();
          break;

        case L1Sensor::WorkMode::Idle:
        default:
          stopAimSession();
          break;
      }
    }

    ++frame_index;
    /******************************* debug ********************************/
    if (overlay_enabled) {
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
  const std::optional<L3Estimation::TrackedTarget>& target,
  L3Estimation::TrackState track_state,
  const L4Planning::Plan& plan,
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
  }
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

}  // namespace
/********************************** debug **********************************/
