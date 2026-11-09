#include "runtime/pose_detection_schedule.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

int main() {
  // Deterministic clock: pose arrives at 6 ms, inference takes 7 ms.
  // Conservative tracked mode requires pose first; acquisition and guarded
  // speculative tracking can detect first. Both share the original deadline.
  for (bool needs_pose : {false, true}) {
    for (int arrival : {0, 6, 12, 30}) {
      int now = 0;
      bool resolved = false;
      auto result = runtime::detectWithPose(needs_pose, [&] {
        now = std::max(now, std::min(arrival, 20)); // original absolute deadline
        resolved = true;
      }, [&] {
        if (resolved != needs_pose) throw std::runtime_error("wrong dependency order");
        now += 7;
        return 42;
      });
      const int expected = needs_pose ? std::min(arrival, 20) + 7
                                      : std::max(std::min(arrival, 20), 7);
      if (!resolved || result != 42 || now != expected)
        throw std::runtime_error("pose deadline or detection result changed");
    }
  }
  std::cout << "pose detection schedule: ready, delayed and timeout cases passed\n";
}
