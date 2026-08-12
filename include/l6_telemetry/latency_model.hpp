#pragma once

namespace L6Telemetry {

struct LatencyModel {
  double capture_ms = 0.0;
  double detect_ms = 0.0;
  double estimate_ms = 0.0;
  double plan_ms = 0.0;
};

}  // namespace L6Telemetry
