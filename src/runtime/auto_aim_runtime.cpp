#include "runtime/auto_aim_runtime.hpp"
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l2_perception/armor.hpp"
#include "l2_perception/armor/armor_detector.hpp"
#include "l2_perception/inference/backends/openvino_backend.hpp"
#include "l3_estimation/pnp_solver.hpp"
#include "l6_telemetry/fps_counter.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"
#include "runtime/auto_aim_config.hpp"
#include <opencv2/opencv.hpp>

#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace {

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
    // 这份 armor.xml 沿用 FosuVision 的 BGR 输入契约；不要在后端交换红蓝通道。
    model_config.model_color_order = L2Perception::ModelColorOrder::Bgr;
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

  //配置并启动串口
  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    L6Telemetry::logWarn("Failed to start serial worker.");
  }
  // PnP 使用启动时加载的静态内参、畸变、枪管外参和装甲板尺寸参数。
    // 标定缺失时 runtime 继续运行检测和显示，
    // 但后续不得生成有效瞄准/开火命令。
  AutoAimConfig auto_aim_config;
  std::optional<L3Estimation::PnpSolver> pnp_solver;
  const auto& camera_calibration = camera->calibration();
  if (!camera_calibration) {
    L6Telemetry::logWarn("PnP disabled: camera calibration is missing");
  } else {
    pnp_solver.emplace(*camera_calibration, auto_aim_config.armor);
    if (pnp_solver->ready()) {
      L6Telemetry::logInfo("PnP solver configured");
    } else {
      L6Telemetry::logWarn("PnP disabled: calibration or armor config is invalid");
    }
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
        // 图像曝光时刻的枪管系 -> 世界系旋转；
        // 当前模型忽略两坐标系原点平移。
        const auto q_world_barrel = serial.gimbalPoseAt(timestamp);
        if (pnp_solver) {
          pnp_solver->set_R_world_barrel(q_world_barrel);
        }
        switch (state.mode) {
        case L1Sensor::WorkMode::AutoAim:
        case L1Sensor::WorkMode::Outpost: {
          // 装甲板检测 → PnP → Tracker → Planner → FireDecision
          auto armors = armor_detector.detect(frame);
          // 保留敌方装甲板
          std::erase_if(armors, [&state](const auto &armor) {
            return !isEnemyArmor(armor.color, state.enemy_color);
          });
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
