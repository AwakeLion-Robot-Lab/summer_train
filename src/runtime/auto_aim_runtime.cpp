#include "runtime/auto_aim_runtime.hpp"
#include <opencv2/opencv.hpp>
#include "l1_sensor/camera/camera.hpp"
#include "l1_sensor/serial/serial_config.hpp"
#include "l1_sensor/serial/serial_worker.hpp"
#include "l6_telemetry/fps_counter.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

#include <memory>
#include <mutex>

namespace runtime {

AutoAimRuntime::AutoAimRuntime(const std::string& config_path)
  : config_path_(config_path)
{
}

void AutoAimRuntime::run()
{
  running_ = true;
  auto camera = std::make_shared<L1Sensor::Camera>(config_path_);
  {
    std::lock_guard<std::mutex> lock(camera_mutex_);
    active_camera_ = camera;
  }
  L6Telemetry::FpsCounter fps_counter;

  //配置并启动串口
  auto serial_config = L1Sensor::loadSerialConfig("config/serial_config.yaml");
  L1Sensor::SerialWorker serial(serial_config);
  const bool serial_started = serial.start();
  if (!serial_started && serial_config.enable) {
    L6Telemetry::logWarn("Failed to start serial worker.");
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
        const auto& state = *robot_state;
        const auto gimbal_pose = serial.gimbalPoseAt(timestamp);
        (void)gimbal_pose;

        switch (state.mode) {
          case L1Sensor::WorkMode::Idle:
            // 不跑视觉控制；发送安全命令；必要时重置 tracker
            break;

          case L1Sensor::WorkMode::AutoAim:
            // 装甲板检测 → PnP → Tracker → Planner → FireDecision
            break;

          case L1Sensor::WorkMode::SmallBuff:
            // 小符专用检测 → PnP → Tracker → Planner → FireDecision
            break;

          case L1Sensor::WorkMode::BigBuff:
            // 大符专用模型与预测参数
            break;

          case L1Sensor::WorkMode::Outpost:
            // 仍属于装甲板链路，但使用前哨站目标模型、跟踪/弹道/火控参数
            break;
        }
      }
    }

    //DEBUG_MODE
    if (false) {
      //帧率
      const double previous_fps = fps_counter.fps();
      const double fps = fps_counter.update();
      if (fps > 0.0 && fps != previous_fps) 
      {
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
