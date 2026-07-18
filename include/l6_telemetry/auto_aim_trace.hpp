#pragma once

#include "l2_perception/armor.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"
#include "l5_control/serial_command.hpp"
#include "l6_telemetry/frame_trace.hpp"
#include "l6_telemetry/latency_model.hpp"

#include <optional>
#include <vector>

namespace L6Telemetry {

// 一帧自瞄闭环的完整旁路记录。这里只保存数据，
// 不在高频链路中执行序列化。
struct AimTrace {
  FrameTrace frame;
  std::vector<L2Perception::Armor> detections;
  std::vector<L3Estimation::Armor> armors;
  std::optional<L3Estimation::Target> target;
  L4Planning::Plan plan;
  L5Control::FireDecision fire;
  L5Control::SerialCommand command;
  LatencyModel latency;

  // 用于验证一帧只 predict 一次、correct 最多一次。
  int predict_count{0};
  int correct_count{0};
};

}  // namespace L6Telemetry
