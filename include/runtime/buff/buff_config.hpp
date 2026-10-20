#pragma once

#include "l2_perception/buff/buff_detector.hpp"
#include "l3_estimation/buff/types.hpp"
#include "l4_planning/buff/types.hpp"
#include "l5_control/buff/controller.hpp"

#include <string>

namespace runtime::buff {

struct Config {
  L2Perception::Buff::DetectorConfig detector;
  L3Estimation::Buff::Config tracker;
  L4Planning::Buff::Config planner;
  L5Control::Buff::Config control;
  bool overlay{false};
  bool plot{false};
};

[[nodiscard]] Config loadConfig(const std::string& path);

}  // namespace runtime::buff
