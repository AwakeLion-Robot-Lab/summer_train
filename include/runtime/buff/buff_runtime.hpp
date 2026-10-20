#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"
#include "l1_sensor/serial/robot_state.hpp"
#include "l2_perception/buff/buff_detector.hpp"
#include "l3_estimation/buff/tracker.hpp"
#include "l4_planning/buff/planner.hpp"
#include "l5_control/buff/controller.hpp"
#include "runtime/buff/buff_config.hpp"

#include <Eigen/Geometry>
#include <chrono>
#include <memory>
#include <optional>
#include <opencv2/core/mat.hpp>

namespace runtime::buff {

struct Result {
  L2Perception::Buff::Detections detections;
  L3Estimation::Buff::Addition addition;
  // tracker 本帧的状态快照。L6 遥测要导出转速/相位/正弦拟合，不能只靠
  // plan 里的 yaw/pitch；丢失时为 nullopt，与 command=safeHold 对应。
  std::optional<L3Estimation::Buff::State> state;
  L4Planning::Buff::Plan plan;
  std::optional<L5Control::SerialCommand> command;
};

class Runtime {
public:
  Runtime(const L1Sensor::CameraCalibration& calibration, Config config);

  [[nodiscard]] L2Perception::Buff::Detections detect(
    const cv::Mat& image, L1Sensor::EnemyColor color);

  [[nodiscard]] Result update(
    L2Perception::Buff::Detections detections,
    const Eigen::Quaterniond& q_world_camera,
    const Eigen::Vector3d& t_world_camera,
    std::chrono::steady_clock::time_point image_timestamp,
    std::chrono::steady_clock::time_point plan_time,
    double reported_bullet_speed);

  [[nodiscard]] std::optional<L5Control::SerialCommand> safeHold() const;
  void reset() noexcept;

private:
  Config config_;
  L2Perception::Buff::Detector detector_;
  L3Estimation::Buff::Tracker tracker_;
  L4Planning::Buff::Planner planner_;
  L5Control::Buff::Controller controller_;
};

}  // namespace runtime::buff
