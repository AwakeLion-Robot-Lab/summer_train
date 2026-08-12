#include "l4_planning/planner.hpp"

#include "l4_planning/ballistic_solver.hpp"
#include "l6_telemetry/math.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace L4Planning {
namespace {

[[nodiscard]] Plan rejected(PlanError error, TimePoint plan_time)
{
  Plan plan;
  plan.plan_time = plan_time;
  plan.error = error;
  return plan;
}

}  // namespace

Planner::Planner(PlanConfig config)
: config_(std::move(config))
{
}

Plan Planner::plan(
  const std::optional<L3Estimation::TrackedTarget>& target,
  const L1Sensor::RobotState& robot_state,
  TimePoint plan_time,
  bool to_now)
{
  PlanInput input;
  input.target = target;
  input.robot_state = robot_state;
  input.plan_time = plan_time;
  input.to_now = to_now;
  return plan(input);
}

Planner::AimPoint Planner::chooseAimPoint(const L3Estimation::TrackedTarget& target)
{
  const L3Estimation::TargetStateVector& state = target.state();
  const std::vector<Eigen::Vector4d> armors = target.armorPoses();
  if (armors.empty()) {
    return {};
  }

  // delta_angle：装甲板法线方向与"整车中心 -> 枪口"方向的夹角。为 0 表示这块
  // 板正对枪口，绝对值越大越斜，命中面积越小。
  const double center_yaw = std::atan2(
    state[L3Estimation::CenterY], state[L3Estimation::CenterX]);
  std::vector<double> delta_angles;
  delta_angles.reserve(armors.size());
  for (const Eigen::Vector4d& armor : armors) {
    delta_angles.push_back(L6Telemetry::limit_rad(armor[3] - center_yaw));
  }

  const auto pointAt = [&](int armor_id) {
    AimPoint point;
    point.valid = true;
    point.armor_id = armor_id;
    point.xyza = armors[static_cast<std::size_t>(armor_id)];
    point.delta_angle = delta_angles[static_cast<std::size_t>(armor_id)];
    return point;
  };

  // 没跳过板时，整车 yaw 还没被数据约束过，只有当前观测到的这块板可信。
  if (!target.jumped) {
    return pointAt(0);
  }

  if (target.name == L3Estimation::ArmorName::Outpost) {
    // 前哨站匀速转，一侧的板不断转入、另一侧不断转出，打转入侧命中率更高。
    for (std::size_t id = 0; id < armors.size(); ++id) {
      const double delta = delta_angles[id];
      if (std::abs(delta) > config_.selector.outpost_coming_angle) {
        continue;
      }
      const double leaving = config_.selector.outpost_leaving_angle;
      if ((state[L3Estimation::Vyaw] > 0.0 && delta < leaving) ||
          (state[L3Estimation::Vyaw] < 0.0 && delta > -leaving)) {
        return pointAt(static_cast<int>(id));
      }
    }
    return {};
  }

  // 普通车辆：取窗口内最正对枪口的板。
  std::vector<int> ids;
  ids.reserve(armors.size());
  for (std::size_t id = 0; id < armors.size(); ++id) {
    if (std::abs(delta_angles[id]) <= config_.selector.front_window) {
      ids.push_back(static_cast<int>(id));
    }
  }
  if (ids.empty()) {
    return {};
  }

  // 两块板都在窗口内（各约 45 度）时，两者的 delta_angle 会反复交替变大变小，
  // 每帧重选就会让瞄准点在两块板之间横跳，所以锁住先选中的那块，直到只剩一块
  // 板在窗口内为止。
  if (ids.size() > 1) {
    const int id0 = ids[0];
    const int id1 = ids[1];
    if (locked_id_ != id0 && locked_id_ != id1) {
      locked_id_ = std::abs(delta_angles[static_cast<std::size_t>(id0)]) <
                       std::abs(delta_angles[static_cast<std::size_t>(id1)])
                     ? id0
                     : id1;
    }
    return pointAt(locked_id_);
  }

  locked_id_ = -1;
  return pointAt(ids[0]);
}

Plan Planner::plan(const PlanInput& input)
{
  if (!input.target.has_value()) {
    return rejected(PlanError::NoTarget, input.plan_time);
  }
  if (!input.target->valid()) {
    return rejected(PlanError::NoArmor, input.plan_time);
  }

  L3Estimation::TrackedTarget target = *input.target;

  // 转得快的目标，指令要更早发出去；这一档差别是实车标出来的。
  const double delay_time = target.state()[L3Estimation::Vyaw] > config_.decision_speed
                              ? config_.high_speed_delay_time
                              : config_.low_speed_delay_time;

  // 弹速无效时仍然解算并让云台跟随，只是这一帧不许开火。
  double bullet_speed = input.robot_state.bullet_speed;
  const bool bullet_speed_ok = config_.bulletSpeedValid(bullet_speed);
  if (!bullet_speed_ok) {
    bullet_speed = config_.fallback_bullet_speed;
  }

  Delay delay;
  delay.image_to_plan =
    input.to_now ? L6Telemetry::delta_time(input.plan_time, target.timestamp()) : 0.005;
  delay.control_to_fire = delay_time;

  // 第一步：外推到弹丸出膛的时刻。
  const TimePoint fire_time =
    target.timestamp() + L6Telemetry::toDuration(delay.beforeFire());
  target.predict(fire_time);

  AimPoint aim = chooseAimPoint(target);
  if (!aim.valid) {
    return rejected(PlanError::OutOfWindow, input.plan_time);
  }

  auto solve = [&](const AimPoint& point) {
    const Eigen::Vector3d xyz = point.xyza.head<3>();
    return solveBallistic(std::hypot(xyz.x(), xyz.y()), xyz.z(), bullet_speed, config_.ballistic);
  };

  std::optional<Ballistic> ballistic = solve(aim);
  if (!ballistic) {
    return rejected(PlanError::BallisticFailed, input.plan_time);
  }

  // 第二步：飞行时间和命中点互相依赖，迭代到飞行时间收敛。每轮都从同一个出膛
  // 时刻的副本重新外推，不能在上一轮的结果上继续累加。
  const double tolerance = std::chrono::duration<double>(config_.fly_time_tolerance).count();
  double previous_fly_time = ballistic->fly_time;
  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    L3Estimation::TrackedTarget hit_target = target;
    hit_target.predict(fire_time + L6Telemetry::toDuration(previous_fly_time));

    aim = chooseAimPoint(hit_target);
    if (!aim.valid) {
      return rejected(PlanError::OutOfWindow, input.plan_time);
    }

    ballistic = solve(aim);
    if (!ballistic) {
      return rejected(PlanError::BallisticFailed, input.plan_time);
    }
    if (std::abs(ballistic->fly_time - previous_fly_time) < tolerance) {
      break;
    }
    previous_fly_time = ballistic->fly_time;
  }

  delay.fire_to_hit = ballistic->fly_time;

  Plan plan;
  plan.target_id = static_cast<int>(target.name);
  plan.armor_id = aim.armor_id;
  plan.plan_time = input.plan_time;
  plan.fire_time = fire_time;
  plan.hit_time = fire_time + L6Telemetry::toDuration(ballistic->fly_time);
  plan.aim_point = aim.xyza.head<3>();
  plan.yaw = std::atan2(plan.aim_point.y(), plan.aim_point.x()) + config_.yaw_offset;
  // 世界系里抬头为负，所以弹道解出的抬升角要取反。
  plan.pitch = -(ballistic->pitch + config_.pitch_offset);
  plan.fly_time = ballistic->fly_time;
  plan.delay = delay;
  plan.ballistic_valid = true;
  plan.fire_armor_id = aim.armor_id;
  plan.fire_delta_angle = aim.delta_angle;
  plan.fire_armor_point = aim.xyza.head<3>();
  plan.fire_admissible = bullet_speed_ok;
  plan.error = bullet_speed_ok ? PlanError::None : PlanError::BadBulletSpeed;
  plan.valid = std::isfinite(plan.yaw) && std::isfinite(plan.pitch);

  if (!plan.valid) {
    return rejected(PlanError::BallisticFailed, input.plan_time);
  }
  return plan;
}

}  // namespace L4Planning
