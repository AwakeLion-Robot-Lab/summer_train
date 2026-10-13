#pragma once

#include <utility>
#include <vector>

/**
 * @file
 * @brief 观测关联的通用件，装甲板和符都用
 */
namespace L3Estimation {

/**
 * @brief 贪心匹配
 *
 * 每轮扫一遍代价矩阵，取还没被占用的组合里代价最小、且小于 max_cost 的一对，
 * 把这两行列都标记为已用，直到再也找不出这样的一对。
 *
 * 不用匈牙利算法是有意的：候选规模最多 3 块板乘几个检测，贪心与最优解几乎
 * 总是一致；而贪心可能锁死次优对的失败模式，被 max_cost 门限兜住了。
 *
 * @tparam CostMatrix 支持 cost[i][j] 取值的二维容器
 * @param cost 代价矩阵，行为观测、列为候选
 * @param observation_count 观测个数
 * @param candidate_count 候选个数
 * @param max_cost 代价门限，严格小于它才能配对
 * @return (观测下标, 候选下标) 的配对表，顺序即锁定顺序
 */
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

}  // namespace L3Estimation
