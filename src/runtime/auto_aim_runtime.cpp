#include "runtime/auto_aim_runtime.hpp"
#include <opencv2/opencv.hpp>
#include "l1_sensor/camera.hpp"
#include "l6_telemetry/fps_counter.hpp"
#include "l6_telemetry/logger.hpp"
#include "l6_telemetry/math.hpp"

namespace runtime {

AutoAimRuntime::AutoAimRuntime(const std::string& config_path)
  : config_path_(config_path)
{
}

void AutoAimRuntime::run()
{
  running_ = true;
  L1Sensor::Camera camera(config_path_);
  L6Telemetry::FpsCounter fps_counter;
  
  cv::namedWindow("auto_aim", cv::WINDOW_NORMAL);

  cv::Mat frame;
  std::chrono::steady_clock::time_point timestamp;

  while (running_) {
    //获取相机帧和时间辍
    camera.read(frame, timestamp);
    if (frame.empty()) {
      L6Telemetry::logWarn("Failed to read frame from camera.");
      continue;
    }
    //DEBUG_MODE
    if(true){
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

  cv::destroyWindow("auto_aim");
}

void AutoAimRuntime::stop()
{
  running_ = false;
}

}  // namespace runtime
