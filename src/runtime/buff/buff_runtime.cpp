#include "runtime/buff/buff_runtime.hpp"

#include <chrono>
#include <utility>

namespace runtime::buff {

Runtime::Runtime(const L1Sensor::CameraCalibration& calibration, Config config)
: config_(config), detector_(config.detector), tracker_(calibration, config.tracker),
  planner_(config.planner), controller_(config.control)
{
  detector_.setCalibration(calibration);
}

L2Perception::Buff::Detections Runtime::detect(
  const cv::Mat& image, L1Sensor::EnemyColor color)
{
  detector_.setEnemyColor(color);
  return detector_.detect(image);
}

Result Runtime::update(
  L2Perception::Buff::Detections detections,
  const Eigen::Quaterniond& q_world_camera,
  const Eigen::Vector3d& t_world_camera,
  std::chrono::steady_clock::time_point image_timestamp,
  std::chrono::steady_clock::time_point plan_time,
  double reported_bullet_speed)
{
  Result result;
  result.detections = std::move(detections);
  const auto bullet_speed =
    config_.planner.resolveBulletSpeed(reported_bullet_speed);
  result.plan.bullet_speed = bullet_speed;
  const auto state = tracker_.update(
    result.detections.icons, result.detections.bullseyes,
    q_world_camera, t_world_camera, image_timestamp);
  result.addition = tracker_.addition();
  result.state = state;
  if (!state) {
    result.command = controller_.safeHold();
    return result;
  }

  // 与源 RuneComponent 一致：当前可用 Trackable 超过 100 ms 就只保持角度，
  // 不继续发送 shoot=true。Tracker 自身的 1.5 s 超时负责彻底销毁模型。
  const auto age = plan_time - tracker_.lastTimestamp();
  if (age > std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>{0.1})) {
    result.plan.degraded = true;
    result.command = controller_.safeHold();
    return result;
  }

  result.plan = planner_.plan(*state, plan_time, bullet_speed, true);
  result.command = controller_.update(result.plan);
  return result;
}

std::optional<L5Control::SerialCommand> Runtime::safeHold() const
{
  return controller_.safeHold();
}

void Runtime::reset() noexcept
{
  tracker_.reset();
  planner_.reset();
  controller_.reset();
}

}  // namespace runtime::buff
