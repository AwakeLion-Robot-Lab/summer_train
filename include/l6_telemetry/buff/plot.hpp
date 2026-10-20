#pragma once

#include "l3_estimation/buff/types.hpp"
#include "l4_planning/buff/types.hpp"

#include <chrono>
#include <string>

namespace L6Telemetry::Buff {

struct PlotSample {
  std::chrono::steady_clock::time_point timestamp{};
  L3Estimation::Buff::State state{};
  L4Planning::Buff::Plan plan{};
};

// 轻量 CSV 行格式，供现有 UDP/日志适配层接入；不引入新的线程或全局状态。
[[nodiscard]] std::string csvHeader();
[[nodiscard]] std::string toCsv(const PlotSample& sample);

}  // namespace L6Telemetry::Buff
