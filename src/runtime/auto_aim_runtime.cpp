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
#include "runtime/auto_aim_config.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

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
    // Decoder 的字段布局跟着 model_path 走，同样来自 inference 节点。
    return L2Perception::ArmorDetector(std::move(backend), config.decoder);
  } catch (const std::exception& error) {
    // 模型或 SDK 不可用时只在启动阶段记录一次；空 Detector 会持续返回安全的空结果。
    L6Telemetry::logError(
      "armor model unavailable",
      std::string{L2Perception::inferenceBackendName(config.inference_backend)},
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

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;
  // "规划结束 -> 串口发出"的实测耗时。本帧的值要等规划做完才知道，所以
  // 用上一帧的量代入本帧的延迟链；这一段帧间基本恒定。
  double measured_plan_to_send = 0.0;

  // 主循环是**单线程同步**的，这是设计选择而不是待办事项：自瞄的代价函数是
  // 开火那一刻的位置误差，不是帧率。异步/流水线推理换来的是吞吐，代价是结果
  // 相对曝光时刻多滞后一帧——这一整帧都会计入 L4Planning::Delay 的
  // image_to_plan，再被 v_yaw 放大成瞄准偏差，得不偿失。
  // 真正降低单帧延迟的并行在推理内部（auto_aim.yaml 的 num_threads /
  // scheduling_core_type），那一层已经开着。
  //
  // 只有 SerialWorker 自带 rx/tx 线程，因为串口是独立的 IO 时序。
  // tools/LatesBuffer 的 LatestBuffer（单槽、新帧覆盖旧帧、统计丢帧）是为
  // 将来可能拆出的取图线程预留的，当前管线不使用它——拆取图线程只在
  // detect 稳定快于帧周期时才有收益，否则只会让处理的帧越来越旧。
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
      switch (state->mode) {
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

          // 叠加层画在命令下发之后，不占用瞄准链路的时间预算。
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
          measured_plan_to_send = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - plan_time).count();
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
