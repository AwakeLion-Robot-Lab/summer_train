#include "l4_planning/latency_compensator.hpp"

#include "l4_planning/types.hpp"

#include <cmath>
#include <utility>

namespace L4Planning {

LatencyCompensator::LatencyCompensator(LatencyConfig config)
  : config_(std::move(config))
{
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
