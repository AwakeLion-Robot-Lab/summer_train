#include "l4_planning/latency_compensator.hpp"

#include <chrono>
#include <cmath>
#include <limits>

//正常情况计算错误，程序返回 1；
//非法输入被错误接受，程序返回 2；
//全部正确则返回 0，表示测试通过。

//目前测试没有打印详细错误信息
int main()
{
  using namespace std::chrono_literals;

  const auto camera_time = L4Planning::TimePoint{100ms};//相机时间辍
  const auto command_time = camera_time + 6ms;//命令时间辍
  L4Planning::LatencyConfig config;
  config.high_speed_fire_delay = 0.030;
  config.low_speed_fire_delay = 0.015;
  config.decision_speed = 8.0;
  const L4Planning::LatencyCompensator compensator{config};

  const auto result = compensator.calculate(camera_time, command_time);
  if (!result.valid ||
      std::abs(result.delay.beforeFire() - 0.006) > 1e-12 ||
      std::abs(result.delay.total() - 0.021) > 1e-12) {
    return 1;
  }

  const auto high_speed =
    compensator.calculate(camera_time, command_time, 8.1);
  const auto negative_high_speed =
    compensator.calculate(camera_time, command_time, -9.0);
  if (!high_speed.valid || !negative_high_speed.valid
      || std::abs(high_speed.delay.fire_delay - 0.030) > 1e-12
      || std::abs(negative_high_speed.delay.fire_delay - 0.015) > 1e-12) {
    return 4;
  }

  if (compensator.calculate({{}, command_time, 0.010}).valid ||
      compensator.calculate({command_time, camera_time, 0.010}).valid ||
      compensator.calculate({camera_time, command_time, -0.001}).valid ||
      compensator.calculate(
        {camera_time, command_time,
         std::numeric_limits<double>::quiet_NaN()}).valid ||
      compensator.calculate(
        {camera_time, command_time,
         std::numeric_limits<double>::infinity()}).valid) {
    return 2;
  }

  L4Planning::LatencyConfig invalid_latency = config;
  invalid_latency.low_speed_fire_delay = -0.001;
  const L4Planning::LatencyCompensator invalid_config{invalid_latency};
  if (invalid_config.calculate(camera_time, command_time).valid) {
    return 3;
  }

  return 0;
}
