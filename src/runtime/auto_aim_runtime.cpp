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
#include "l6_telemetry/logger.hpp"
#include "runtime/armor_detector_factory.hpp"
#include "runtime/auto_aim_config.hpp"
#include "runtime/auto_aim_debug.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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
    LOG_ERROR(
      "armor model or side-light model unavailable",
      std::string{L2Perception::backendName(config.inference_backend)},
      config.model_path.string(), error.what());
    return {};
  }
}

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
  // 启动时只加载一次模型；每帧仅执行预处理、推理、解码和数字分类。
  L2Perception::ArmorDetector armor_detector =
    loadDetector(auto_aim_config);

  //配置并启动串口
  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    LOG_WARN("Failed to start serial worker.");
  }
  // L3 EskfTracker 持有 PnP（仅整车初始化用）和 IESKF。标定缺失时 runtime 继续运行检测和显示，
  // 但后续不得生成有效瞄准/开火命令。
  std::optional<L3Estimation::EskfTracker> tracker;
  const auto& camera_calibration = camera->calibration();
  if (!camera_calibration) {
    LOG_WARN("Tracker disabled: camera calibration is missing");
  } else {
    tracker.emplace(
      *camera_calibration,
      auto_aim_config.armor,
      auto_aim_config.ieskf_tracker,
      auto_aim_config.ieskf_target);
    if (tracker->ready()) {
      LOG_INFO("L3 tracker configured");
    } else {
      LOG_WARN(
        "Tracker disabled: calibration, armor, or tracker config is invalid");
    }
  }

  L4Planning::Planner planner(auto_aim_config.plan);
  L5Control::Controller controller(auto_aim_config.fire);

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

  // 叠加层、遥测、强制 WorkMode 都在 AutoAimDebug 里，按 debug: 各自开关。
  AutoAimDebug debug(auto_aim_config, camera_calibration);
  const auto forced_mode = debug.forcedMode();
  // 只用来在模式切换时打一条日志：进自瞄前滤波器是否已在跟，要靠它对时间。
  std::optional<L1Sensor::WorkMode> last_mode;

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
      // 覆盖只改分派用的模式，state 本身不动——日志仍然反映下位机真正上报的
      // 值，否则排查时会看不出电控到底给没给对模式。
      const L1Sensor::WorkMode mode = forced_mode.value_or(state->mode);
      if (mode != last_mode) {
        LOG_INFO(
          "work mode ->", L1Sensor::toString(mode), "| tracker",
          tracker ? L3Estimation::toString(tracker->state()) : "disabled");
        last_mode = mode;
      }
      switch (mode) {
        // Idle 也照跑 L2/L3，只是不出命令。电控切进自瞄的那一刻，滤波器若
        // 已经跟着目标，就不必从零收敛：冷启动回放里速度要十几到几十帧才稳，
        // 小陀螺反转时更久。ROI 也照开，否则一进自瞄网络输入从整图换成裁剪，
        // 检出的板宽差约 4.5%，深度跳变会被当成径向速度。
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost:
        case L1Sensor::WorkMode::Idle: {
          const bool aiming = mode != L1Sensor::WorkMode::Idle;
          // 同 sp_vision：先等图像之后的那包姿态到齐，再前后两包插值。图像
          // 刚到手时它往往还在路上，直接查只能拿最新一包顶替，云台一转就
          // 错开几度。只等一个发包周期；等满说明串口断流，退回最新一包。
          const auto wait_start = std::chrono::steady_clock::now();
          const bool pose_ready = serial.waitPose(timestamp);
          const double pose_wait_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
          if (!pose_ready) {
            LOG_DEBUG("gimbal pose wait timeout");
          }
          const auto image_pose = serial.gimbalPoseAt(timestamp);

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
              tracker->lightRoi(image_pose, timestamp, frame.size());
            light_hints = tracker->lightHints(image_pose, timestamp);
            net_roi = tracker->netFocusRoi(
              image_pose, timestamp, frame.size(),
              armor_detector.net_aspect_ratio());
          }
          // 侧边灯条按下位机给的敌方颜色筛：传 Unknown 会把友军灯条也送进
          // L3 关联。装甲板在这里按同一颜色过滤。
          const auto detect = [&](const std::optional<cv::Rect>& roi) {
            auto result = armor_detector.detectFrame(
              frame, light_roi, roi, enemyArmorColor(state->enemy_color),
              light_hints);
            std::erase_if(result.armors, [&state](const auto& armor) {
              return !isEnemyArmor(armor.color, state->enemy_color);
            });
            return result;
          };
          auto perception = detect(net_roi);
          // 冷启动那一帧（ieskf.init_roi）：在初始化会挑的那块板周围按跟踪时
          // 的 ROI 再检一次，初始化和之后的 ROI 帧同一尺度。ROI 里一块都没检
          // 出就退回整图结果。
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

          L4Planning::Plan plan;
          L5Control::FireDecision fire;
          std::optional<std::chrono::steady_clock::time_point> plan_time;
          std::optional<Eigen::Quaterniond> actual_pose;
          std::optional<L5Control::SerialCommand> command;
          if (aiming) {
            // 规划与开火判定使用推理结束时刻。
            plan_time = std::chrono::steady_clock::now();
            actual_pose = serial.gimbalPoseAt(*plan_time);

            // L4: 预测命中时刻、选板并解算弹道。
            plan = planner.plan({
              .target = target,
              .bullet_speed = state->bullet_speed,
              .plan_time = *plan_time,
              .to_now = true,
              .plan_to_send = measured_plan_to_send,
              .q_world_barrel = actual_pose});

            // L5: 开火判定、命令跳变检查和安全保持。
            command = controller.update(target, track_state, plan, actual_pose);
            fire = controller.lastDecision();

            // L1: 下发 L5 生成的控制命令，并量出本帧规划到发送的耗时，
            // 供下一帧的延迟链使用。必须紧接着 updateCommand 取，放到遥测和
            // 叠加层之后会把它们的几毫秒算进 plan_to_send。
            if (command) {
              serial.updateCommand(*command);
            }
            measured_plan_to_send = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - *plan_time).count();
          } else {
            // Idle 不出命令，跟踪器不动。reset 让进自瞄后的头一次选板挑离枪口
            // 最近的板，不让第一条命令甩到车的另一侧。
            planner.reset();
            sendSafeHold();
          }

          // 遥测与叠加层都在命令下发之后，不占用瞄准链路的时间预算。
          debug.record(frame, {
            .timestamp = timestamp,
            .plan_time = plan_time,
            .image_pose = image_pose,
            .actual_pose = actual_pose,
            .aiming = aiming,
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
            .command = &command});
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

    if (!debug.show(frame)) {
      running_ = false;
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
