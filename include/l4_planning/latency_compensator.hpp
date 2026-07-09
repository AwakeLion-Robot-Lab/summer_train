#pragma once

namespace L4Planning {

class LatencyCompensator {
public:
  explicit LatencyCompensator(double latency_seconds = 0.0);
  double latency() const;

private:
  double latency_seconds_ = 0.0;
};

}  // namespace L4Planning
