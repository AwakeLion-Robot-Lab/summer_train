#include "runtime/auto_aim_runtime.hpp"

#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/target_estimator.hpp"
#include "l4_planning/planner.hpp"
#include "l4_planning/planner_config.hpp"
#include "l5_control/controller.hpp"
#include "l6_telemetry/fps_counter.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace {

[[maybe_unused]] bool isEnemyArmor(L2Perception::ArmorColor observed,
                                   L1Sensor::EnemyColor expected) noexcept
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

L2Perception::ArmorDetector makeArmorDetector()
{
  const std::filesystem::path model_path{"model/armor_model/armor.xml"};

  try {
    auto backend = std::make_unique<L2Perception::OpenVinoBackend>();
    L2Perception::InferenceModelConfig model_config;
    model_config.model_path = model_path;
    model_config.device = "CPU";
    model_config.model_color_order = L2Perception::ModelColorOrder::Rgb;
    model_config.normalization_divisor = 255.0F;
    backend->load(model_config);

    L6Telemetry::logInfo("armor model loaded", model_path.string(),
                         model_config.device);
    return L2Perception::ArmorDetector(std::move(backend));
  } catch (const std::exception& error) {
    L6Telemetry::logError("armor model unavailable", model_path.string(),
                          error.what());
    return {};
  }
}

std::optional<L3Estimation::TargetState> chooseTarget(
    const std::vector<L3Estimation::TargetState>& targets)
{
  if (targets.empty()) {
    return std::nullopt;
  }
  return targets.front();
}

}  // namespace

namespace runtime {

AutoAimRuntime::AutoAimRuntime(const std::string& config_path)
    : config_path_(config_path)
{
}

void AutoAimRuntime::run()
{
  running_ = true;

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

  L6Telemetry::FpsCounter fps_counter;
  L2Perception::ArmorDetector armor_detector = makeArmorDetector();

  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    L6Telemetry::logWarn("Failed to start serial worker.");
  }

  std::optional<L3Estimation::TargetEstimator> target_estimator;
  if (const auto calibration = camera->calibration()) {
    const auto gimbal_pose_provider =
      [&serial](L3Estimation::TimePoint timestamp) {
        return serial.gimbalPoseAt(timestamp);
      };

    try {
      target_estimator.emplace(
        *calibration,
        Eigen::Matrix3d::Identity(),
        Eigen::Vector3d::Zero(),
        gimbal_pose_provider);
    } catch (const std::exception& error) {
      L6Telemetry::logWarn("Failed to create target estimator", error.what());
    }
  } else {
    L6Telemetry::logWarn("camera calibration missing; L3/L4 disabled");
  }

  L4Planning::Planner planner(planner_tuning.planner);
  L5Control::Controller controller;
  L4Planning::PlannerContext planner_context;
  planner_context.config = planner_tuning.planner;
  planner_context.latency = planner_tuning.latency;
  planner_context.armor_score_weights = planner_tuning.armor_score_weights;
  planner_context.facing_angle_good = planner_tuning.facing_angle_good;
  planner_context.facing_angle_bad = planner_tuning.facing_angle_bad;

  cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;

  while (running_) {
    if (!camera->read(frame, timestamp)) {
      if (!running_) {
        break;
      }
      continue;
    }

    if (serial_started) {
      const auto robot_state = serial.latestState();
      if (robot_state && target_estimator) {
        const auto& state = *robot_state;
        switch (state.mode) {
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost: {
          auto armors = armor_detector.detect(frame);
          std::erase_if(armors, [&state](const auto& armor) {
            return !isEnemyArmor(armor.color, state.enemy_color);
          });

          const auto targets = target_estimator->update(armors, timestamp);
          const auto target = chooseTarget(targets);
          if (target) {
            planner_context.planning_time = timestamp;
            const auto plan = planner.plan(target, state, planner_context);
            if (plan.valid) {
              serial.updateCommand(controller.makeCommand(plan));
            }
          }
          break;
        }

        case L1Sensor::WorkMode::SmallBuff:
        case L1Sensor::WorkMode::BigBuff:
        case L1Sensor::WorkMode::Idle:
        default:
          break;
        }
      }
    }

    if (false) {
      const double previous_fps = fps_counter.fps();
      const double fps = fps_counter.update();
      if (fps > 0.0 && fps != previous_fps) {
        L6Telemetry::logDebug("auto_aim fps", fps);
      }
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

void AutoAimRuntime::stop()
{
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

}  // namespace runtime
