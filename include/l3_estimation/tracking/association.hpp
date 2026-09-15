#pragma once

#include "l3_estimation/types.hpp"

#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

// 观测关联与跟踪状态机的通用件。照搬 awakening 的 dta_utils.hpp:18-102。
namespace L3Estimation {

// 贪心匹配：每轮取全局最小代价的一对，锁定后从候选中划掉，直到没有可用对。
//
// 不用匈牙利算法是有意的：候选规模最多 3 块板 × 几个检测，贪心与最优解几乎
// 总是一致，而贪心的失败模式（把一个次优对锁死）在这里由门控兜住——超过
// 门限的代价根本进不了矩阵。
template <typename CostMatrix>
std::vector<std::pair<int, int>> greedyMatch(
  const CostMatrix & cost, int observation_count, int candidate_count, double max_cost)
{
  std::vector<bool> observation_used(observation_count, false);
  std::vector<bool> candidate_used(candidate_count, false);
  std::vector<std::pair<int, int>> result;

  while (true) {
    double best = max_cost;
    int best_observation = -1;
    int best_candidate = -1;

    for (int observation = 0; observation < observation_count; ++observation) {
      if (observation_used[observation]) {
        continue;
      }
      for (int candidate = 0; candidate < candidate_count; ++candidate) {
        if (candidate_used[candidate]) {
          continue;
        }
        if (cost[observation][candidate] < best) {
          best = cost[observation][candidate];
          best_observation = observation;
          best_candidate = candidate;
        }
      }
    }

    if (best_observation < 0 || best_candidate < 0) {
      break;
    }
    observation_used[best_observation] = true;
    candidate_used[best_candidate] = true;
    result.emplace_back(best_observation, best_candidate);
  }

  return result;
}

// 四态跟踪机的状态与计数。
struct TrackLifecycle
{
  TrackState state{TrackState::Lost};
  int detect_count{0};
  int lost_count{0};

  bool isTracking() const noexcept
  {
    return state == TrackState::Tracking || state == TrackState::TempLost;
  }

  void reset() noexcept
  {
    state = TrackState::Lost;
    detect_count = 0;
    lost_count = 0;
  }
};

// 推进状态机。found 表示本帧是否成功关联到观测。
//
// Detecting 要连续 tracking_thres 帧才转 Tracking，中途丢一帧就退回 Lost——
// 这条严格性是防误初始化的：一次偶然的错误关联不该建立一个目标。而 Tracking
// 丢失后先进 TempLost 靠预测撑住，超时才放弃。
inline void updateFsm(
  bool found, TrackLifecycle & lifecycle, int tracking_thres, double lost_time,
  double lost_time_thres)
{
  switch (lifecycle.state) {
    case TrackState::Detecting:
      if (!found) {
        lifecycle.reset();
        break;
      }
      if (++lifecycle.detect_count > tracking_thres) {
        lifecycle.state = TrackState::Tracking;
        lifecycle.detect_count = 0;
      }
      break;

    case TrackState::Tracking:
      if (!found) {
        lifecycle.state = TrackState::TempLost;
        lifecycle.lost_count = 0;
      }
      break;

    case TrackState::TempLost:
      if (found) {
        lifecycle.state = TrackState::Tracking;
        lifecycle.lost_count = 0;
        break;
      }
      if (lost_time > lost_time_thres) {
        lifecycle.reset();
      }
      break;

    case TrackState::Lost:
      break;
  }
}

inline double elapsedSeconds(const TimePoint & from, const TimePoint & to) noexcept
{
  return std::max(0.0, std::chrono::duration<double>(to - from).count());
}

}  // namespace L3Estimation
