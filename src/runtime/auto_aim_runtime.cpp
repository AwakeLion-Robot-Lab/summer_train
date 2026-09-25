#include "runtime/auto_aim_runtime.hpp"
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/armor/eskf_tracker.hpp"
#include "l4_planning/armor/planner.hpp"
#include "l5_control/controller.hpp"
#include "l6_telemetry/aim_overlay.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "l6_telemetry/udp_json_sender.hpp"
#include "runtime/armor_detector_factory.hpp"
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

namespace {

// L1 的 enemy_color 是下位机给出的目标阵营；L2 的 ArmorColor 是图像识别结果。
// 两个枚举同序，所以下位机的值直接转过来用，不必逐项映射；static_assert
// 保证将来任一侧增改成员时编译期就炸，而不是把 Blue 静默当成 Red。
static_assert(
  static_cast<int>(L1Sensor::EnemyColor::Red) ==
      static_cast<int>(L2Perception::ArmorColor::Red) &&
    static_cast<int>(L1Sensor::EnemyColor::Blue) ==
      static_cast<int>(L2Perception::ArmorColor::Blue) &&
    static_cast<int>(L1Sensor::EnemyColor::Unknown) ==
      static_cast<int>(L2Perception::ArmorColor::Unknown),
  "EnemyColor 与 ArmorColor 必须同序");

constexpr L2Perception::ArmorColor enemyArmorColor(
  L1Sensor::EnemyColor enemy) noexcept
{
  return static_cast<L2Perception::ArmorColor>(enemy);
}

// 识别出的颜色与下位机给的阵营一致即为敌方。Unknown 要单独挡掉：两侧都会
// 用它表示"没有有效值"，放过去就等于允许误击友军。
constexpr bool isEnemyArmor(
  L2Perception::ArmorColor observed, L1Sensor::EnemyColor expected) noexcept
{
  return expected != L1Sensor::EnemyColor::Unknown &&
         observed == enemyArmorColor(expected);
}

L2Perception::ArmorDetector loadDetector(const runtime::AutoAimConfig& config)
{
  try {
    return runtime::makeDetector(config);
  } catch (const std::exception& error) {
    // 模型或 SDK 不可用时只在启动阶段记录一次；空 Detector 会持续返回安全的空结果。
    L6Telemetry::logError(
      "armor model or side-light model unavailable",
      std::string{L2Perception::backendName(config.inference_backend)},
      config.model_path.string(), error.what());
    return {};
  }
}

/********************************** debug **********************************/
// 遥测只服务调试，不参与瞄准链路，定义在文件末尾，整段删掉也不影响自瞄。
struct TelemetryInput {
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point plan_time;
  std::optional<Eigen::Quaterniond> image_pose;
  std::optional<Eigen::Quaterniond> actual_pose;
  bool pose_ready{false};
  double pose_wait_ms{0.0};
  const L1Sensor::RobotState* state{nullptr};
  const L1Sensor::SerialWorker* serial{nullptr};
  const L2Perception::ArmorFrame* perception{nullptr};
  const L3Estimation::EskfTracker* tracker{nullptr};
  const std::optional<L3Estimation::EskfTarget>* target{nullptr};
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  const L4Planning::Plan* plan{nullptr};
  const L5Control::FireDecision* fire{nullptr};
  const std::optional<L5Control::SerialCommand>* command{nullptr};
};

nlohmann::json telemetryFrame(const TelemetryInput& in);
/********************************** debug **********************************/

} // namespace

namespace runtime {

AutoAimRuntime::AutoAimRuntime(const std::string &config_path)
    : config_path_(config_path) {}

void AutoAimRuntime::run() {
  running_ = true;
  const AutoAimConfig auto_aim_config =
    loadConfig("config/auto_aim.yaml");
  auto camera = std::make_shared<L1Sensor::Camera>(config_path_);
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    active_camera_ = camera;
  }
  // 启动时只加载一次模型；每帧仅执行预处理、推理、解码和角点精修。
  L2Perception::ArmorDetector armor_detector =
    loadDetector(auto_aim_config);

  //配置并启动串口
  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    L6Telemetry::logWarn("Failed to start serial worker.");
  }
  // L3 EskfTracker 持有 PnP（仅整车初始化用）和 IESKF。标定缺失时 runtime 继续运行检测和显示，
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

  // 曲线遥测与叠加层各自独立开关：实车上没有显示器，要的恰好是曲线。
  // UDP 是无连接的，没人接收也不会阻塞或报错。
  std::optional<L6Telemetry::UdpJsonSender> plotter;
  if (auto_aim_config.debug.plot) {
    plotter.emplace(
      auto_aim_config.debug.plot_host,
      static_cast<std::uint16_t>(auto_aim_config.debug.plot_port));
    L6Telemetry::logInfo(
      "telemetry enabled", plotter->host(), std::to_string(plotter->port()));
  }

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;
  // "规划结束 -> 串口发出"的实测耗时。本帧的值要等规划做完才知道，所以
  // 用上一帧的量代入本帧的延迟链；这一段帧间基本恒定。
  double measured_plan_to_send = 0.0;

  // 单线程同步是设计选择不是待办：自瞄的代价是开火那一刻的位置误差而不是
  // 帧率，异步流水线换来吞吐、代价是结果多滞后一帧，那一帧会进
  // Delay::image_to_plan 再被 v_yaw 放大成瞄准偏差。真正降低单帧延迟的并行
  // 在推理内部（num_threads / scheduling_core_type），那层已经开着。
  // tools/LatesBuffer 是给将来的取图线程预留的，当前管线不用。
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
      switch (L1Sensor::WorkMode::AutoAim) {
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost: {
          // 曝光时刻的姿态要晚 pose_delay_ms 才到，此刻还没收到。ROI 只是
          // 裁剪范围，先用手上最新的姿态；滤波器用的精确姿态等检测做完再取，
          // 等待与检测重叠，不白白加延迟。
          const auto roi_pose = serial.gimbalPoseAt(timestamp);

          // L2: ROI 聚焦 + 检测，保留敌方装甲板。两个 ROI 都由上一帧的整车
          // 状态外推到本帧曝光时刻：net_roi 喂灯条模型，远距小目标裁剪后再
          // resize 相当于局部放大；light_roi 决定哪些灯条作为独立观测交给 L3，
          // 越紧越不容易把别的车的灯条混进来。Lost/冷启动时前者退化为整图、
          // 后者返回空，等价于全图检测、不给独立灯条。
          std::optional<cv::Rect> light_roi;
          std::optional<cv::Rect> net_roi;
          std::vector<L2Perception::LightHint> light_hints;
          if (tracker && tracker->ready()) {
            light_roi =
              tracker->lightRoi(roi_pose, timestamp, frame.size());
            light_hints = tracker->lightHints(roi_pose, timestamp);
            net_roi = tracker->netFocusRoi(
              roi_pose, timestamp, frame.size(),
              armor_detector.net_aspect_ratio());
          }
          // 侧边灯条按下位机给的敌方颜色筛：传 Unknown 会把友军灯条也送进
          // L3 关联。装甲板在下面按同一颜色过滤。
          auto perception = armor_detector.detectFrame(
            frame, light_roi, net_roi, enemyArmorColor(state->enemy_color),
            light_hints);
          std::erase_if(perception.armors, [&state](const auto& armor) {
            return !isEnemyArmor(armor.color, state->enemy_color);
          });

          // 等曝光时刻的姿态到齐再更新滤波器。等不到（串口断流）就用边界值，
          // 遥测里 pose_ready=0 会标出来。
          const auto wait_start = std::chrono::steady_clock::now();
          const bool pose_ready = serial.waitPose(timestamp);
          const double pose_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
          const auto image_pose = serial.gimbalPoseAt(timestamp);

          // L3: 关联、状态机与整车 IESKF。独立灯条与装甲板角点一起进
          // updateMulti()，每根灯条都是一个四维端点观测，独立灯条是
          // 在完整板之外多出来的那部分信息。
          std::optional<L3Estimation::EskfTarget> target;
          if (tracker && tracker->ready()) {
            target = tracker->track(
              perception.armors, perception.lights, image_pose, timestamp);
          }
          const auto track_state = tracker
            ? tracker->state()
            : L3Estimation::TrackState::Lost;

          // 规划与开火判定使用推理结束时刻。
          const auto plan_time = std::chrono::steady_clock::now();
          const auto actual_pose = serial.gimbalPoseAt(plan_time);

          // L4: 预测命中时刻、选板并解算弹道。
          L4Planning::PlanInput plan_input;
          plan_input.target = target;
          plan_input.robot_state = *state;
          plan_input.plan_time = plan_time;
          plan_input.to_now = true;
          plan_input.plan_to_send = measured_plan_to_send;
          const auto plan = planner.plan(plan_input);

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

          // 规划到发送的实测耗时必须在 updateCommand 之后立刻取，遥测和画图
          // 的耗时不能算进 plan_to_send。
          measured_plan_to_send = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - plan_time).count();

          // 遥测与叠加层都在命令下发之后，不占用瞄准链路的时间预算。
          if (plotter) {
            const auto fire = controller.lastDecision();
            (void)plotter->send(telemetryFrame({
              .timestamp = timestamp,
              .plan_time = plan_time,
              .image_pose = image_pose,
              .actual_pose = actual_pose,
              .pose_ready = pose_ready,
              .pose_wait_ms = pose_wait_ms,
              .state = &*state,
              .serial = &serial,
              .perception = &perception,
              .tracker = tracker ? &*tracker : nullptr,
              .target = &target,
              .track_state = track_state,
              .plan = &plan,
              .fire = &fire,
              .command = &command}));
          }
          if (overlay_solver && tracker &&
              frame_index % auto_aim_config.debug.overlay_every == 0) {
            overlay_solver->set_R_world_barrel(image_pose);
            L6Telemetry::drawAimOverlay(
              frame,
              {.detections = perception.armors,
               .target = target,
               .track_state = track_state,
               .plan = plan,
               .fire = controller.lastDecision(),
               .q_world_barrel = image_pose},
              *overlay_solver, *camera_calibration);
          }
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
    if (overlay_enabled) {
      cv::imshow("auto_aim", frame);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        running_ = false;
      }
    }
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
namespace {

// PlotJuggler 遥测。分节点是硬要求：曝光时刻云台角 gimbal/、规划时刻云台角
// gimbal_now/、真正下发的命令 cmd/ 必须落在不同分支上，三者同图才看得出
// 跟随误差和振荡；观测计数 track/ 与滤波状态 ekf/ 同理。角度 degree、时间 ms。
nlohmann::json telemetryFrame(const TelemetryInput& in)
{
  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  const auto ms = [](auto duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  };
  const auto putYawPitch = [&](nlohmann::json& node, const Eigen::Quaterniond& q) {
    const Eigen::Vector3d ypr = L6Telemetry::eulers(q.toRotationMatrix(), 2, 1, 0);
    node["yaw"] = ypr[0] * kRadToDeg;
    node["pitch"] = ypr[1] * kRadToDeg;
  };
  nlohmann::json data;

  // 帧曝光时刻，单位 s。必须发：主循环周期本身在抖，按 UDP 到达时刻排点会把
  // 平滑曲线画成台阶。在 PlotJuggler 的 UDP/JSON 插件里选它作 timestamp。
  data["t"] = std::chrono::duration<double>(in.timestamp.time_since_epoch()).count();

  if (in.image_pose) {
    putYawPitch(data["gimbal"], *in.image_pose);
  }
  if (in.actual_pose) {
    putYawPitch(data["gimbal_now"], *in.actual_pose);
  }
  data["gimbal"]["bullet_speed"] = in.state->bullet_speed;

  // cmd: 本帧真正交给串口的命令（含规划失败时的 safeHold），振荡看这条。
  if (*in.command) {
    data["cmd"]["yaw"] = (*in.command)->yaw * kRadToDeg;
    data["cmd"]["pitch"] = (*in.command)->pitch * kRadToDeg;
    data["cmd"]["shoot"] = (*in.command)->shoot ? 1 : 0;
  }
  data["cmd"]["sent"] = in.command->has_value() ? 1 : 0;

  // serial: img_minus_imu 是图像曝光时刻减最新一帧姿态的接收时刻。为正说明
  // 图像比最新姿态还新、gimbalPoseAt 只能取边界值不做插值；呈锯齿说明串口
  // 批量收包，图像与姿态实际没对齐。云台一动，这个误差就变成假的目标运动。
  data["serial"]["img_minus_imu"] = ms(in.timestamp - in.state->timestamp);
  data["serial"]["plan_minus_imu"] = ms(in.plan_time - in.state->timestamp);
  // pose_wait 是检测做完后还要等姿态的时间；pose_ready=0 说明等满了也没等到。
  data["serial"]["pose_ready"] = in.pose_ready ? 1 : 0;
  data["serial"]["pose_wait"] = in.pose_wait_ms;
  data["serial"]["rx"] = in.serial->receivedStateCount();
  data["serial"]["rx_dropped"] = in.serial->droppedPacketCount();
  data["serial"]["tx"] = in.serial->sentCommandCount();
  data["serial"]["tx_failed"] = in.serial->failedCommandCount();

  // track: 状态机与本帧观测量。drops 是单调计数，单帧重建在状态图上看不见，
  // 在它上面一定留下台阶。
  data["track"]["state"] = static_cast<int>(in.track_state);
  data["track"]["n_armors"] = static_cast<int>(in.perception->armors.size());
  data["track"]["n_lights"] = static_cast<int>(in.perception->lights.size());
  if (in.tracker) {
    data["track"]["n_used_lights"] = static_cast<int>(in.tracker->usedLights().size());
    data["track"]["match_count"] = in.tracker->lastMatchCount();
    data["track"]["drops"] = static_cast<std::uint64_t>(in.tracker->dropCount());
  }

  // ekf: 整车 13 维，按 VehicleModel::idx 的顺序命名。
  if (*in.target) {
    namespace idx = L3Estimation::VehicleModel::idx;
    const auto& target = **in.target;
    const Eigen::VectorXd x = target.ekf_x();
    auto& ekf = data["ekf"];
    ekf["x"] = x[idx::CX];
    ekf["vx"] = x[idx::VCX];
    ekf["y"] = x[idx::CY];
    ekf["vy"] = x[idx::VCY];
    ekf["z"] = x[idx::CZ];
    ekf["vz"] = x[idx::VCZ];
    ekf["yaw"] = x[idx::ROT_Z] * kRadToDeg;
    ekf["w"] = x[idx::VYAW];
    ekf["r1"] = x[idx::LOG_R1];
    ekf["p1"] = x[idx::P1];
    ekf["p2"] = x[idx::P2];
    ekf["tilt_pitch"] = x[idx::ROT_Y] * kRadToDeg;
    ekf["tilt_roll"] = x[idx::ROT_X] * kRadToDeg;
    ekf["last_id"] = target.last_id;
    ekf["jumped"] = target.jumped ? 1 : 0;
    ekf["nis"] = target.lastNis();
    ekf["nis_dof"] = target.lastNisDof();
  }

  // aim: L4 规划出的云台目标角，和 cmd/ 的差别只在规划失败的帧。
  const L4Planning::Plan& plan = *in.plan;
  data["aim"]["status"] = static_cast<int>(plan.status);
  data["aim"]["reason"] = static_cast<int>(plan.reason);
  if (plan.valid()) {
    data["aim"]["yaw"] = plan.aim.yaw * kRadToDeg;
    data["aim"]["pitch"] = plan.aim.pitch * kRadToDeg;
  }
  data["aim"]["armor_id"] = plan.fire ? plan.fire->armor_id : -1;

  // delay: 五段延迟链分开发，绝不合并成一个标量。
  const L4Planning::Delay& delay = plan.timing.delay;
  data["delay"]["image_to_plan"] = delay.image_to_plan * 1e3;
  data["delay"]["plan_to_send"] = delay.plan_to_send * 1e3;
  data["delay"]["send_to_control"] = delay.send_to_control * 1e3;
  data["delay"]["control_to_fire"] = delay.control_to_fire * 1e3;
  data["delay"]["fire_to_hit"] = delay.fire_to_hit * 1e3;
  data["delay"]["before_fire"] = delay.beforeFire() * 1e3;

  const L5Control::FireDecision& fire = *in.fire;
  data["fire"]["feasible"] = fire.fire_feasible ? 1 : 0;
  data["fire"]["shoot"] = fire.shoot ? 1 : 0;
  data["fire"]["yaw_err"] = fire.yaw_error * kRadToDeg;
  data["fire"]["pitch_err"] = fire.pitch_error * kRadToDeg;
  data["fire"]["reason"] =
    fire.reasons.empty() ? -1 : static_cast<int>(fire.reasons.front());

  return data;
}

}  // namespace
/********************************** debug **********************************/
