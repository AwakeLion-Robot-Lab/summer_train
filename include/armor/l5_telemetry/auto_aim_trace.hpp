#pragma once

#include "l1_perception/armor.hpp"
#include "l2_estimation/armor/eskf_target.hpp"
#include "l3_planning/types.hpp"
#include "l4_control/fire_decision.hpp"
#include "l4_control/serial_command.hpp"
#include "l5_telemetry/frame_trace.hpp"
#include "l5_telemetry/latency_model.hpp"

#include <optional>
#include <vector>

namespace L6Telemetry {

// 一帧自瞄闭环的完整旁路记录。这里只保存数据，
// 不在高频链路中执行序列化。
struct AimTrace {
  FrameTrace frame;
  std::vector<L2Perception::Armor> detections;
  std::vector<L3Estimation::Armor> armors;
  std::optional<L3Estimation::EskfTarget> target;
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  L4Planning::AimPlan plan;
  L5Control::FireDecision fire;
  L5Control::SerialCommand command;
  LatencyModel latency;

  // 用于验证一帧只 predict 一次、correct 最多一次。
  int predict_count{0};
  int correct_count{0};
};

}  // namespace L6Telemetry
