#include "l4_planning/latency_compensator.hpp"

#include "l4_planning/types.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace L4Planning {

double LatencyConfig::fireDelay(double target_yaw_rate) const noexcept
{
  if (!ready() || !std::isfinite(target_yaw_rate)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return target_yaw_rate > decision_speed
    ? high_speed_fire_delay
    : low_speed_fire_delay;
}

LatencyCompensator::LatencyCompensator(LatencyConfig config)
  : config_(std::move(config))
{
}

LatencyResult LatencyCompensator::calculate(
  TimePoint camera_timestamp,
  TimePoint command_timestamp) const noexcept
{
  return calculate(camera_timestamp, command_timestamp, 0.0);
}

LatencyResult LatencyCompensator::calculate(
  TimePoint camera_timestamp,
  TimePoint command_timestamp,
  double target_yaw_rate) const noexcept
{
  if (!config_.ready()) {
    return {};
  }
  return calculate(
    Delay{
      camera_timestamp,
      command_timestamp,
      config_.fireDelay(target_yaw_rate)});
}

LatencyResult LatencyCompensator::calculate(const Delay& delay) const noexcept
{
  LatencyResult result;
  if (!config_.ready() ||
      delay.camera_timestamp == TimePoint{} ||
      delay.command_timestamp < delay.camera_timestamp ||
      !std::isfinite(delay.fire_delay) ||
      delay.fire_delay < 0.0) {
    return result;
  }

  // 系统出枪延迟 = 图像到命令发布的耗时 + 命令发布到弹丸出膛的标定耗时。
  // 调用 total() 同时确保聚合结果本身可用，避免异常标定值继续传入预测器。
  const double total_delay = delay.total();
  if (!std::isfinite(total_delay) || total_delay < 0.0) {
    return result;
  }

  result.delay = delay;
  result.valid = true;
  return result;
}

}  // namespace L4Planning
