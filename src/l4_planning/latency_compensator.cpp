#include "l4_planning/latency_compensator.hpp"

#include <algorithm>
#include <chrono>

namespace L4Planning {

LatencyCompensator::LatencyCompensator(PlanConfig config)
: config_(std::move(config))
{
}

Delay LatencyCompensator::measure(TimePoint image_time, TimePoint plan_time) const
{
  Delay delay;

  // 曝光到规划：时间戳倒退说明上游给错了时间，按 0 处理而不是负延迟，
  // 负延迟会让预测倒推、把瞄准点甩到目标身后。
  const double image_to_plan =
    std::chrono::duration<double>(plan_time - image_time).count();
  delay.image_to_plan = std::max(0.0, image_to_plan);

  // plan_to_send 需要下发时刻才能测量，由 L5 回填；这里保持 0。
  delay.plan_to_send = 0.0;

  // 标定段：未标定时保持 0，fireDelayReady() 会拦住开火。
  delay.send_to_control = config_.send_to_control.value_or(0.0);
  delay.control_to_fire = config_.control_to_fire.value_or(0.0);

  // fire_to_hit 由弹道解算填入。
  delay.fire_to_hit = 0.0;
  return delay;
}

}  // namespace L4Planning
