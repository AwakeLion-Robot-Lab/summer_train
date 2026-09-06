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
  const L4Planning::LatencyCompensator compensator{{0.010}};//弹丸出射延迟

  const auto result = compensator.calculate(camera_time, command_time);
  if (!result.valid ||
      std::abs(result.delay.beforeFire() - 0.006) > 1e-12 ||
      std::abs(result.delay.total() - 0.016) > 1e-12) {
    return 1;
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

  const L4Planning::LatencyCompensator invalid_config{{-0.001}};
  if (invalid_config.calculate(camera_time, command_time).valid) {
    return 3;
  }

  return 0;
}
