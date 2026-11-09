#include "l3_planning/tiny_mpc.hpp"
#include "l3_planning/armor_candidate_generator.hpp"
#include "l3_planning/in_plate_aim.hpp"
#include "l3_planning/planner.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace L4Planning {
namespace {

TimePoint addSeconds(TimePoint time, double seconds)
{
  return time + std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(seconds));
}

// 求当前 MPC 指令的真实弹道与运动板面的交点，不能仅用线性化角度盒判断。
bool resolveCommandHit(AimPlan& plan, const L3Estimation::TargetState& target,
  const L1Sensor::RobotState& robot, const Eigen::Isometry3d& transform,
  const PlannerConfig& config)
{
  const Eigen::Vector3d initial_velocity = robot.bullet_speed * Eigen::Vector3d{
    std::cos(plan.pitch) * std::cos(plan.yaw),
    std::cos(plan.pitch) * std::sin(plan.yaw), std::sin(plan.pitch)};
  const auto evaluate = [&](double time, PredictionResult& prediction,
                            ArmorPose& armor, Eigen::Vector3d& relative) {
    prediction = Predictor{}.predict({target, addSeconds(plan.fire_time, time)});
    if (!prediction.valid) return false;
    const auto found = std::find_if(prediction.armor_candidates.begin(), prediction.armor_candidates.end(),
      [&](const ArmorPose& p) { return p.armor_id == plan.armor_id && p.valid; });
    if (found == prediction.armor_candidates.end()) return false;
    armor = *found;
    double travel = time, drop = 0.5 * config.gravity * time * time;
    if (config.enable_air_resistance) {
      const double k = config.linear_drag_coefficient;
      travel = -std::expm1(-k * time) / k;
      drop = config.gravity * (time - travel) / k;
    }
    relative = initial_velocity * travel - Eigen::Vector3d{0.0, 0.0, drop}
      - transform.translation() - armor.position_world;
    return relative.allFinite();
  };
  double time = plan.fly_time;
  PredictionResult prediction;
  ArmorPose armor;
  Eigen::Vector3d relative;
  bool converged = false;
  for (int i = 0; i < 20; ++i) {
    if (!evaluate(time, prediction, armor, relative)) return false;
    const double distance = armor.rotation_world.col(0).dot(relative);
    if (std::abs(distance) < 1e-7) { converged = true; break; }
    PredictionResult next_prediction;
    ArmorPose next_armor;
    Eigen::Vector3d next_relative;
    if (!evaluate(time + 1e-4, next_prediction, next_armor, next_relative)) return false;
    const double derivative = (next_armor.rotation_world.col(0).dot(next_relative) - distance) / 1e-4;
    if (!std::isfinite(derivative) || derivative <= 1e-3) return false;
    time -= std::clamp(distance / derivative, -0.03, 0.03);
    if (time <= 0.0 || time > 2.0 * plan.fly_time + 0.1) return false;
  }
  if (!converged) return false;
  // 只允许从可见面进入，拒绝背面交点。
  PredictionResult after_prediction;
  ArmorPose after_armor;
  Eigen::Vector3d after_relative;
  if (!evaluate(time + 1e-4, after_prediction, after_armor, after_relative)
      || (after_armor.rotation_world.col(0).dot(after_relative)
        - armor.rotation_world.col(0).dot(relative)) / 1e-4 <= 1e-3) return false;
  auto aim = queryInPlateAim(prediction, armor, plan.aim_query_state, plan.fire_time,
    robot.rpy.yaw, config.in_plate_aim, -transform.translation());
  if (!aim.valid) return false;
  aim.offset = armor.rotation_world.rightCols<2>().transpose() * relative;
  if ((aim.offset.array().abs() > aim.limits.array() + 1e-7).any()) return false;
  aim.offset = aim.offset.cwiseMax(-aim.limits).cwiseMin(aim.limits);
  aim.position_world = armor.position_world + armor.rotation_world.rightCols<2>() * aim.offset;
  const double dt = std::chrono::duration<double>(plan.fire_time - plan.aim_query_state.timestamp).count();
  if (plan.aim_query_state.armor_id == plan.armor_id && dt > 1e-6)
    aim.velocity = (aim.offset - plan.aim_query_state.offset) / dt;
  plan.in_plate_aim = aim;
  plan.aim_point_world = aim.position_world;
  plan.aim_point_barrel = aim.position_world + transform.translation();
  plan.armor_center_world = armor.position_world;
  plan.armor_center_barrel = armor.position_world + transform.translation();
  plan.armor_rotation_world = armor.rotation_world;
  plan.fly_time = time;
  plan.impact_time = addSeconds(plan.fire_time, time);
  // 命中时刻变化后重算窗口，仍保持已选板号和跨帧锁板状态。
  PlannerContext context;
  context.config = config;
  context.T_barrel_world = transform;
  const auto candidate = generateArmorCandidates(target, robot, context, plan.fire_time,
    plan.aim_query_state, plan.armor_id, aim.offset);
  if (!candidate || candidate->empty() || !candidate->front().valid) return false;
  plan.within_firing_window = candidate->front().within_firing_window;
  return true;
}

}  // namespace

AimPlan applyTinyMpc(
  const AimPlan& direct_plan,
  const L3Estimation::TargetState& target,
  const L1Sensor::RobotState& robot_state,
  const Eigen::Isometry3d& T_barrel_world,
  const PlannerConfig& config)
{
  AimPlan result = direct_plan;
  result.using_MPC = false;
  result.samples.clear();
  if (!config.enable_mpc || !direct_plan.valid) {
    return result;
  }

  TinyMpcReference reference = buildTinyMpcReference(
    direct_plan, target, robot_state, T_barrel_world, config);
  TinyMpcSolution solution = solveTinyMpc(reference, config);
  if (reference.optimize_in_plate) {
    // 重新迭代偏移后的飞行时间和预测板姿态，检查局部模型是否收敛。
    bool converged = false;
    for (int iteration = 0; solution.valid && iteration < 3; ++iteration) {
      const auto previous_regions = reference.regions;
      if (!refineTinyMpcPlateReference(reference, solution, direct_plan, target,
          robot_state, T_barrel_world, config)) return result;
      double angle_error = 0.0, offset_error = 0.0;
      for (std::size_t k = 0; k < kTinyMpcHorizon; ++k) {
        const Eigen::Vector2d linear = previous_regions[k].center
          + previous_regions[k].jacobian * solution.in_plate_offsets[k];
        const Eigen::Vector2d exact{reference.yaw.angle[k], reference.pitch.angle[k]};
        angle_error = std::max(angle_error, (exact - linear).norm());
        offset_error = std::max(offset_error,
          (reference.in_plate_aim[k].offset - solution.in_plate_offsets[k]).norm());
      }
      // 0.1 mrad 模型误差远小于常用几何角度容差；当前执行点仍须通过真实交点检查。
      if (angle_error <= 1e-4 && offset_error <= 1e-5) { converged = true; break; }
      solution = solveTinyMpc(reference, config);
    }
    if (!converged) return result;
  }
  std::vector<AimSample> samples = makeTinyMpcSamples(
    reference, solution, direct_plan.generated_at);
  if (samples.empty()) {
    return result;
  }

  result.samples = std::move(samples);
  result.using_MPC = true;
  const AimSample& current = result.samples.front();
  result.yaw = current.yaw;
  result.pitch = current.pitch;
  result.yaw_rate = current.yaw_rate;
  result.pitch_rate = current.pitch_rate;
  result.yaw_acceleration = current.yaw_acceleration;
  result.pitch_acceleration = current.pitch_acceleration;
  if (reference.optimize_in_plate) {
    // 先保留经过固定点迭代的板内参考；实际指令在板间过渡时仍能输出运动。
    constexpr auto k = kTinyMpcControlIndex;
    result.in_plate_aim = reference.in_plate_aim[k];
    result.aim_point_world = result.in_plate_aim.position_world;
    result.aim_point_barrel = result.aim_point_world + T_barrel_world.translation();
    result.impact_time = reference.impact_time[k];
    result.fly_time = std::chrono::duration<double>(result.impact_time - result.fire_time).count();
    const auto prediction = Predictor{}.predict({target, result.impact_time});
    if (!prediction.valid) return direct_plan;
    const auto armor = std::find_if(prediction.armor_candidates.begin(), prediction.armor_candidates.end(),
      [&](const ArmorPose& p) { return p.armor_id == result.armor_id && p.valid; });
    if (armor == prediction.armor_candidates.end()) return direct_plan;
    result.armor_center_world = armor->position_world;
    result.armor_center_barrel = armor->position_world + T_barrel_world.translation();
    result.armor_rotation_world = armor->rotation_world;
    result.mpc_aim_feasible = resolveCommandHit(result, target, robot_state, T_barrel_world, config);
    result.fire_permitted = result.tracked_ready && result.within_firing_window && result.mpc_aim_feasible;
  }
  return result;
}

}  // namespace L4Planning
