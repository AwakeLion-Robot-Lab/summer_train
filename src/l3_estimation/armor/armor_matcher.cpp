#include "l3_estimation/armor/armor_matcher.hpp"

#include "l3_estimation/tracking/association.hpp"
#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>

namespace L3Estimation {

namespace VM = VehicleModel;

namespace {

// 装甲板四角在图像上的顺序：左上、右上、右下、左下。预测侧也按这个顺序由
// 左右灯条的端点拼出来，两边不一致的话四边形代价算的是两个不同形状。
constexpr int kCorners = 4;

// 代价矩阵里"不可配对"的填充值，比 greedyMatch 的门限大一档。
constexpr double kMaxCost = 1e9;

// 算一块板正对相机的程度：板的 x 轴指向车心，朝外的法向是 -axis_x，它与
// 「板 → 相机」方向的点积越大，板越正对。
double facingScore(const Eigen::Isometry3d & armor_in_camera)
{
  const Eigen::Vector3d front_normal = -armor_in_camera.linear().col(0);
  return front_normal.dot(-armor_in_camera.translation());
}

double segmentAngle(const cv::Point2f & from, const cv::Point2f & to)
{
  return std::atan2(to.y - from.y, to.x - from.x);
}

// 每块板正对相机的程度，下标即板编号。
std::vector<double> facingScores(
  const EskfTarget & target, const Eigen::VectorXd & state, const ObsContext & ctx)
{
  const int count = target.armor_num();
  std::vector<double> facing(count, 0.0);
  for (int id = 0; id < count; ++id) {
    const auto pose_in_world =
      VM::armorPose<double>(state.data(), id, count, target.name);
    facing[id] = facingScore(ctx.camera_in_world.inverse() * pose_in_world);
  }
  return facing;
}

}  // namespace

std::vector<MatchedArmor> matchArmor(
  const EskfTarget & target, const Eigen::VectorXd & state, const ObsContext & ctx,
  const std::vector<Armor> & armors)
{
  std::vector<MatchedArmor> result;
  if (armors.empty() || !target.initialized()) {
    return result;
  }

  const EskfTargetConfig & config = target.config();

  // 可见性筛选：按正对相机的程度排序，只留最正对的三块。
  const std::vector<double> facing = facingScores(target, state, ctx);
  std::vector<std::pair<double, int>> ranked;
  ranked.reserve(facing.size());
  for (int id = 0; id < static_cast<int>(facing.size()); ++id) {
    ranked.emplace_back(facing[id], id);
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
    return a.first > b.first;
  });

  std::vector<int> candidates;
  const std::size_t visible_count = std::min<std::size_t>(3, ranked.size());
  for (std::size_t i = 0; i < visible_count; ++i) {
    candidates.push_back(ranked[i].second);
  }

  const int observation_count = static_cast<int>(armors.size());
  // 关联门限：如果目标刚跳变过，允许更大的代价；否则用较小的门限。
  const double gate = target.jumped ? config.match_gate : config.match_gate_not_all_init;

  std::vector<std::vector<double>> cost(
    observation_count, std::vector<double>(candidates.size(), kMaxCost + 1.0));

  for (int j = 0; j < observation_count; ++j) {
    const std::array<cv::Point2f, kCorners> & measured = armors[j].points;

    for (std::size_t i = 0; i < candidates.size(); ++i) {
      const int id = candidates[i];
      const auto left = ctx.project(id, true, state);
      const auto right = ctx.project(id, false, state);

      // 拼出预测四边形，顺序与检测角点一致：左上、右上、右下、左下。
      const std::array<cv::Point2f, kCorners> predicted_corners{
        left.first, right.first, right.second, left.second};

      // 代价是四边形与四边形的差异，三项各描述一个自由度：
      //   中心误差 —— 整体位置
      //   边角度误差 —— 姿态
      //   周长比例误差 —— 距离缩放
      cv::Point2f predicted_center(0.0F, 0.0F);
      cv::Point2f measured_center(0.0F, 0.0F);
      for (int k = 0; k < kCorners; ++k) {
        predicted_center += predicted_corners[k];
        measured_center += measured[k];
      }
      predicted_center *= 0.25F;
      measured_center *= 0.25F;
      const double center_error = cv::norm(predicted_center - measured_center);

      double angle_error = 0.0;
      double predicted_perimeter = 0.0;
      double measured_perimeter = 0.0;
      for (int k = 0; k < kCorners; ++k) {
        const int next = (k + 1) % kCorners;
        angle_error += std::abs(L6Telemetry::limit_rad(
          segmentAngle(predicted_corners[k], predicted_corners[next]) -
          segmentAngle(measured[k], measured[next])));
        predicted_perimeter += cv::norm(predicted_corners[k] - predicted_corners[next]);
        measured_perimeter += cv::norm(measured[k] - measured[next]);
      }
      const double side_length_error =
        predicted_perimeter > 1e-6
          ? std::abs(predicted_perimeter - measured_perimeter) / predicted_perimeter
          : kMaxCost;

      const double total = config.weight_center_error * center_error +
                           config.weight_angle_error * angle_error +
                           config.weight_side_length_error * side_length_error;

      if (std::isfinite(total) && total < gate) {
        cost[j][i] = total;
      }
    }
  }

  for (const auto & [observation, candidate] :
       greedyMatch(cost, observation_count, static_cast<int>(candidates.size()), kMaxCost)) {
    result.emplace_back(candidates[candidate], armors[observation]);
  }
  return result;
}

std::vector<std::pair<int, bool>> lightSlots(
  const EskfTarget & target, const Eigen::VectorXd & state, const ObsContext & ctx,
  const std::vector<MatchedArmor> & matched_armors)
{
  std::vector<std::pair<int, bool>> slots;
  const int count = target.armor_num();
  if (count <= 0) {
    return slots;
  }
  const std::vector<double> facing = facingScores(target, state, ctx);
  const int closest_id =
    static_cast<int>(std::max_element(facing.begin(), facing.end()) - facing.begin());

  const auto matchedPlate = [&matched_armors](int id) {
    return std::any_of(
      matched_armors.begin(), matched_armors.end(),
      [id](const MatchedArmor & matched) { return matched.first == id; });
  };
  const auto add = [&](int id, bool is_left) {
    // 已配成完整板的板，两根灯条本帧都由 update() 从板的角点拆出来，不再需要
    // 侧边灯条补位；背对相机的板看不到灯条，槽位留着只会招来错配。
    if (matchedPlate(id) || !(facing[id] > 0.0)) {
      return;
    }
    slots.emplace_back(id, is_left);
  };

  // 候选限定在最正对那块板的左右灯条，加上相邻两块板靠近它的各一根。
  add((closest_id + count - 1) % count, false);
  add((closest_id + 1) % count, true);
  add(closest_id, false);
  add(closest_id, true);
  return slots;
}

std::vector<MatchedLight> matchLight(
  const EskfTarget & target, const Eigen::VectorXd & state, const ObsContext & ctx,
  const std::vector<L2Perception::Light> & lights,
  const std::vector<MatchedArmor> & matched_armors, LightMatchStats * stats)
{
  std::vector<MatchedLight> result;
  const EskfTargetConfig & config = target.config();
  // 本帧至少关联上一块完整板才做：没有完整板做锚，侧边灯条的编号和左右归属
  // 几乎是猜的。基地的板不绕转，整车预测也约束不了它的灯条位置。
  if (
    !config.enable_lights_measure || VM::isBase(target.name) || matched_armors.empty() ||
    lights.empty() || !target.initialized() || !target.hasFilter() ||
    (config.light_match_require_jumped && !target.jumped)) {
    if (stats != nullptr) {
      ++stats->frames_skipped;
    }
    return result;
  }

  using PredictedLight = std::tuple<int, bool, std::pair<cv::Point2f, cv::Point2f>>;
  std::vector<PredictedLight> visible_lights;
  for (const auto & [id, is_left] : lightSlots(target, state, ctx, matched_armors)) {
    visible_lights.emplace_back(id, is_left, ctx.project(id, is_left, state));
  }

  if (visible_lights.empty()) {
    if (stats != nullptr) {
      ++stats->frames_no_candidate;
    }
    return result;
  }

  if (stats != nullptr) {
    stats->slots += visible_lights.size();
  }

  const int observation_count = static_cast<int>(lights.size());
  std::vector<std::vector<double>> cost(
    observation_count, std::vector<double>(visible_lights.size(), kMaxCost + 1.0));

  const auto lightCost = [&](const L2Perception::Light & light,
                             const PredictedLight & candidate) -> double {
    const auto & [id, is_left, endpoints] = candidate;
    const auto & [top, bottom] = endpoints;
    const double predicted_length = cv::norm(top - bottom);
    if (!(predicted_length > 1e-6)) {
      return kMaxCost + 1.0;
    }

    if (stats != nullptr) {
      ++stats->considered;
    }

    const double length_error = std::abs(light.length - predicted_length);
    if (length_error > predicted_length * config.light_match_length_ratio_gate) {
      if (stats != nullptr) {
        ++stats->reject_length;
      }
      return kMaxCost + 1.0;
    }

    // 角度取 atan2(Δx, Δy)，预测和检测同一约定，差值再折回 (-π, π]。
    const double predicted_angle = std::atan2(top.x - bottom.x, top.y - bottom.y);
    const double light_angle =
      std::atan2(light.top.x - light.bottom.x, light.top.y - light.bottom.y);
    const double angle_error =
      std::abs(VM::normalizeAngle(light_angle - predicted_angle));
    if (angle_error > config.light_match_angle_gate) {
      if (stats != nullptr) {
        ++stats->reject_angle;
      }
      return kMaxCost + 1.0;
    }

    // 马氏距离用的 S 与这根灯条若被采纳时真正进更新的那份一致。
    const auto distance = target.mahalanobis(
      target.lightObs(ctx, light.top, light.bottom, id, is_left, true));
    if (!distance) {
      return kMaxCost + 1.0;
    }
    if (!(*distance <= config.light_match_chi2_gate)) {
      if (stats != nullptr) {
        ++stats->reject_chi2;
      }
      return kMaxCost + 1.0;
    }
    if (stats != nullptr) {
      ++stats->passed;
      stats->log_length_ratio += std::log(light.length / predicted_length);
    }
    return *distance;
  };

  for (int observation = 0; observation < observation_count; ++observation) {
    for (std::size_t candidate = 0; candidate < visible_lights.size(); ++candidate) {
      cost[observation][candidate] =
        lightCost(lights[observation], visible_lights[candidate]);
    }
  }

  for (const auto & [observation, candidate] : greedyMatch(
         cost, observation_count, static_cast<int>(visible_lights.size()), kMaxCost)) {
    const auto & [id, is_left, unused] = visible_lights[candidate];
    result.emplace_back(id, is_left, lights[observation]);
  }
  if (stats != nullptr) {
    stats->matched += result.size();
    if (!result.empty()) {
      ++stats->frames_matched;
    }
  }
  return result;
}

}  // namespace L3Estimation
