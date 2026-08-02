#include "runtime/auto_aim_runtime.hpp"
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/planner.hpp"
#include "l5_control/controller.hpp"
#include "l5_control/fire_decision.hpp"
#include "l6_telemetry/fps_counter.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include <opencv2/opencv.hpp>

#include <exception>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace {

bool isFiniteTarget(const L3Estimation::TargetState& target) noexcept
{
  return target.robot_id >= 0 &&
         target.center.allFinite() &&
         target.velocity.allFinite() &&
         std::isfinite(target.yaw) &&
         std::isfinite(target.yaw_rate) &&
         std::isfinite(target.radius) &&
         std::isfinite(target.radius_offset) &&
         std::isfinite(target.height_offset) &&
         target.covariance.allFinite();
}

// L1 的 enemy_color 是下位机给出的目标阵营；L2 的 ArmorColor 是图像识别结果。
// 任何一侧未知时都不允许作为自瞄目标，避免误击友军。
[[maybe_unused]] bool isEnemyArmor(L2Perception::ArmorColor observed,
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

L2Perception::ArmorDetector makeArmorDetector()
{
  const std::filesystem::path model_path{"model/armor_model/armor.xml"};

  try {
    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = model_path;
    model_config.device = "CPU";
    // armor.xml 的宿主输入来自 OpenCV BGR 图像，送入模型前必须转换为 RGB。
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);

    L6Telemetry::logInfo("armor model loaded", model_path.string(), model_config.device);
    return L2Perception::ArmorDetector(std::move(backend));
  } catch (const std::exception& error) {
    // 模型或 SDK 不可用时只在启动阶段记录一次；空 Detector 会持续返回安全的空结果。
    L6Telemetry::logError("armor model unavailable", model_path.string(), error.what());
    return {};
  }
}

} // namespace

namespace runtime {

AutoAimRuntime::AutoAimRuntime(const std::string &config_path)
    : config_path_(config_path) {}

void AutoAimRuntime::run() {
  running_ = true;
  auto camera = std::make_shared<L1Sensor::Camera>(config_path_);
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    active_camera_ = camera;
  }
  L6Telemetry::FpsCounter fps_counter;
  // 启动时只加载一次模型；每帧仅执行预处理、推理和 Decoder。
  L2Perception::ArmorDetector armor_detector = makeArmorDetector();

  // 火控配置只在启动时从 YAML 加载一次，FireEvaluator 在后续每帧复用。
  const L5Control::FireConfig fire_config =
    L5Control::loadFireConfig("config/fire_config.yaml");
  L5Control::FireEvaluator fire_evaluator{fire_config};
  L5Control::Controller controller;
  L4Planning::Planner planner;

  //配置并启动串口
  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    L6Telemetry::logWarn("Failed to start serial worker.");
  }

  std::unique_ptr<L3Estimation::TargetEstimator> target_estimator;
  const auto& camera_calibration = camera->calibration();
  if (camera_calibration && camera_calibration->barrelExtrinsicsReady()) {
    const Eigen::Isometry3d& camera_to_control =
      *camera_calibration->T_barrel_camera;
    target_estimator = std::make_unique<L3Estimation::TargetEstimator>(
      *camera_calibration,
      camera_to_control.rotation(),
      camera_to_control.translation(),
      [&serial](L3Estimation::TimePoint time) {
        return serial.gimbalPoseAt(time);
      });
  } else {
    L6Telemetry::logWarn(
      "auto aim calibration unavailable; L5 will remain fail-closed");
  }
  cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;

  while (running_) {
    //获取相机帧和时间辍
    if (!camera->read(frame, timestamp)) {
      if (!running_) {
        break;
      }
      continue;
    }

    if (serial_started) {
      const auto robot_state = serial.latestState();
      if (robot_state) {
        const auto &state = *robot_state;
        std::optional<L3Estimation::TargetState> target;
        L4Planning::AimPlan plan;
        const bool calibration_ready =
          target_estimator != nullptr &&
          camera_calibration->matchesImageSize(frame.size());

        switch (state.mode) {
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost: {
          // 装甲板检测 → PnP → Tracker → Planner → FireDecision
          auto armors = armor_detector.detect(frame);
          // 保留敌方装甲板
          std::erase_if(armors, [&state](const auto &armor) {
            return !isEnemyArmor(armor.color, state.enemy_color);
          });

          if (calibration_ready) {
            const auto targets = target_estimator->update(armors, timestamp);
            // 正式的多目标选择策略尚未实现；多于一个候选时保持关火，
            // 避免 Runtime 在 L4 之外擅自引入目标优先级。
            if (targets.size() == 1 && isFiniteTarget(targets.front())) {
              target = targets.front();
            }
            plan = planner.plan(target, state);
          }
          break;
        }

        case L1Sensor::WorkMode::SmallBuff:
          // 小符专用检测 → PnP → Tracker → Planner → FireDecision
          break;

        case L1Sensor::WorkMode::BigBuff:
          // 大符专用模型与预测参数
          break;

        case L1Sensor::WorkMode::Idle:
          //待机
          break;

        default:
          // 待机
          break;
        }


        const L5Control::FireInput fire_input{
          .target = target,
          .plan = plan,
          .robot_state = state,
          .now = std::chrono::steady_clock::now(),
          .calibration_ready = calibration_ready};
        const L5Control::FireDecision fire_decision =
          fire_evaluator.evaluate(fire_input);
        const L5Control::SerialCommand command =
          controller.makeCommand(plan, fire_decision, state);
        serial.updateCommand(command);
      }
    }

    // DEBUG_MODE
    if (false) {
      //帧率
      const double previous_fps = fps_counter.fps();
      const double fps = fps_counter.update();
      if (fps > 0.0 && fps != previous_fps) {
        L6Telemetry::logDebug("auto_aim fps", fps);
      }
      //单帧时间差
      auto end_time = std::chrono::steady_clock::now();
      double elapsed = L6Telemetry::delta_time(end_time, timestamp);
      L6Telemetry::logDebug("delta_time", elapsed);
    }

    cv::imshow("auto_aim", frame);
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q' || key == 'Q') {
      running_ = false;
    }
  }

  serial.stop();
  camera->stop();
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    if (active_camera_ == camera) {
      active_camera_.reset();
    }
  }
  cv::destroyWindow("auto_aim");
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
