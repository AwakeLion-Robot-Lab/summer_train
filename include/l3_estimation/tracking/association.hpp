#pragma once

#include "l3_estimation/types.hpp"

#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

// 观测关联与跟踪状态机的通用件，装甲板和符都用这里的函数。
namespace L3Estimation {

// 贪心匹配：每轮扫一遍代价矩阵，取还没被占用的组合里代价最小、且小于
// max_cost 的一对，把这两行列都标记为已用，直到再也找不出这样的一对。
// 返回的是 (观测下标, 候选下标) 的配对表，顺序即锁定顺序。
//
// 不用匈牙利算法是有意的：候选规模最多 3 块板乘几个检测，贪心与最优解几乎
// 总是一致；而贪心可能锁死次优对的失败模式，被 max_cost 门限兜住了。
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

// 状态机的状态与两个计数器，由 updateFsm 维护。
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

// 按本帧是否关联上观测（found）推进一步状态机：
//   Detecting  关联上就累加计数，超过 tracking_thres 转 Tracking；丢一帧
//              直接退回 Lost，重新开始计数；
//   Tracking   丢帧转 TempLost；
//   TempLost   再次关联上就回 Tracking，距上次更新超过 lost_time_thres 秒
//              （由调用方算好传进来的 lost_time）则退回 Lost；
//   Lost       什么都不做，新目标的建立由调用方负责。
//
// Detecting 丢一帧就重来是防误初始化：一次偶然的错误关联不该建立一个目标。
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

// from 到 to 的秒数，钳到非负，避免乱序时间戳算出负的 dt。
inline double elapsedSeconds(const TimePoint & from, const TimePoint & to) noexcept
{
  return std::max(0.0, std::chrono::duration<double>(to - from).count());
}

}  // namespace L3Estimation
