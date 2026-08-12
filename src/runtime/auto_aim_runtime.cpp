#include "runtime/auto_aim_runtime.hpp"

#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/inference_backend.hpp"
#include "l3_estimation/filter_est/tracker.hpp"
#include "l3_estimation/gtsam_est/tracker.hpp"
#include "l3_estimation/tracker.hpp"
#include "l6_telemetry/fps_counter.hpp"
#include "l6_telemetry/logger.hpp"
#include "runtime/auto_aim_config.hpp"

#include <opencv2/opencv.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace {

// L1 的 enemy_color 是下位机给出的目标阵营；L2 的 ArmorColor 是图像识别结果。
// 任何一侧未知时都不允许作为自瞄目标，避免误击友军。
bool isEnemyArmor(
  L2Perception::ArmorColor observed, L1Sensor::EnemyColor expected) noexcept
{
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

L2Perception::ArmorDetector makeArmorDetector(const runtime::AutoAimConfig& config)
{
  try {
    auto backend = L2Perception::makeInferenceBackend(config.inference_backend);
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = config.model_path;
    model_config.device = config.inference_device;
    // 宿主图像是 BGR，转成模型需要的 RGB 和归一化都在后端的预处理图里完成。
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);

    L6Telemetry::logInfo(
      "armor model loaded",
      std::string{L2Perception::inferenceBackendName(config.inference_backend)},
      config.model_path.string(), model_config.device);
    return L2Perception::ArmorDetector(std::move(backend));
  } catch (const std::exception& error) {
    // 模型或 SDK 不可用时只在启动阶段记录一次；空 Detector 会持续返回空结果，
    // 主循环照常运行。
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

  const AutoAimConfig config = loadAutoAimConfig("config/auto_aim.yaml");

  auto camera = std::make_shared<L1Sensor::Camera>(config_path_);
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    active_camera_ = camera;
  }
  L6Telemetry::FpsCounter fps_counter;
  // 启动时只加载一次模型；每帧只做预处理、推理和解码。
  L2Perception::ArmorDetector armor_detector = makeArmorDetector(config);

  // 串口：自己跑收发线程，提供最新机器人状态和按时间插值的云台姿态。
  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    L6Telemetry::logWarn("Failed to start serial worker.");
  }

  // L3 Tracker 持有 PnP 和整车状态估计。估计器后端由 config.estimator 选，
  // 请求一个没编进来的后端会在这里抛出，而不是悄悄回退。标定缺失时继续检测和
  // 显示，但不产生瞄准结果。
  std::unique_ptr<L3Estimation::ITracker> tracker;
  const auto& camera_calibration = camera->calibration();
  if (!camera_calibration) {
    L6Telemetry::logWarn("Tracker disabled: camera calibration is missing");
  } else {
    if (config.estimator == L3Estimation::EstimatorBackend::Gtsam) {
      tracker = std::make_unique<L3Estimation::GtsamEst::Tracker>(
        *camera_calibration, config.armor, config.target, config.gtsam);
    } else {
      tracker = std::make_unique<L3Estimation::FilterEst::Tracker>(
        *camera_calibration, config.armor, config.tracker, config.target, config.filter);
    }
    L6Telemetry::logInfo("estimator backend:", L3Estimation::toString(config.estimator));
    if (!tracker->ready()) {
      L6Telemetry::logWarn("Tracker disabled: camera calibration is invalid");
    }
  }

  cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;

  while (running_) {
    // 一帧的链路：取图 -> 取机器人状态 -> 按工作模式分派 -> 检测 -> 跟踪。
    if (!camera->read(frame, timestamp)) {
      if (!running_) {
        break;
      }
      continue;
    }

    const std::optional<L1Sensor::RobotState> robot_state =
      serial_started ? serial.latestState() : std::optional<L1Sensor::RobotState>{};
    if (robot_state) {
      switch (robot_state->mode) {
      case L1Sensor::WorkMode::AutoAim:
      case L1Sensor::WorkMode::Outpost: {
        auto armors = armor_detector.detect(frame);
        std::erase_if(armors, [&robot_state](const auto &armor) {
          return !isEnemyArmor(armor.color, robot_state->enemy_color);
        });

        // 云台姿态必须取**曝光时刻**的插值结果，不能取"现在"，否则 PnP 会把
        // 一帧的云台运动算进装甲板位姿里。
        const auto q_world_barrel = serial.gimbalPoseAt(timestamp);

        // L2 -> L3 转换、逐板 PnP、状态机和 EKF 更新都在 Tracker 内完成。
        [[maybe_unused]] std::optional<L3Estimation::TrackedTarget> target;
        if (tracker && tracker->ready()) {
          target = tracker->track(armors, q_world_barrel, timestamp);
        }
        // TODO: target -> L4Planning::Planner -> L5Control::FireDecider ->
        // SerialWorker，接线前不下发任何指令。
        break;
      }

      case L1Sensor::WorkMode::SmallBuff:
      case L1Sensor::WorkMode::BigBuff:
        // TODO: 打符专用检测与预测，尚未实现。
        break;

      case L1Sensor::WorkMode::Idle:
      default:
        break;
      }
    }

    fps_counter.update();

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
