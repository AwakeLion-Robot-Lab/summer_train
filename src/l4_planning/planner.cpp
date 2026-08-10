#include "l4_planning/planner.hpp"

#include "l6_telemetry/math.hpp"

#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

namespace L4Planning {
namespace {

constexpr double kGravity = 9.7833;

struct SpTrajectory {
  bool unsolvable{true};
  double fly_time{0.0};
  double pitch{0.0};
};

// 逐式对应 sp_vision tools::Trajectory。SP Aimer 使用的就是这条真空弹道，
// 这里不经过其他阻力模型、业务门限或二次求解。
[[nodiscard]] SpTrajectory solveSpTrajectory(
  double bullet_speed, double distance, double height)
{
  SpTrajectory result;
  const double a =
    kGravity * distance * distance / (2.0 * bullet_speed * bullet_speed);
  const double b = -distance;
  const double c = a + height;
  const double delta = b * b - 4.0 * a * c;
  if (delta < 0.0) {
    return result;
  }

  const double tan_pitch_1 = (-b + std::sqrt(delta)) / (2.0 * a);
  const double tan_pitch_2 = (-b - std::sqrt(delta)) / (2.0 * a);
  const double pitch_1 = std::atan(tan_pitch_1);
  const double pitch_2 = std::atan(tan_pitch_2);
  const double time_1 = distance / (bullet_speed * std::cos(pitch_1));
  const double time_2 = distance / (bullet_speed * std::cos(pitch_2));

  result.unsolvable = false;
  result.pitch = time_1 < time_2 ? pitch_1 : pitch_2;
  result.fly_time = time_1 < time_2 ? time_1 : time_2;
  return result;
}

// SP 在预测时统一先乘 1e6、截断成 int，再构造 microseconds。
[[nodiscard]] std::chrono::microseconds spDuration(double seconds)
{
  return std::chrono::microseconds(static_cast<int>(seconds * 1e6));
}

[[nodiscard]] Plan rejected(PlanError error, TimePoint plan_time)
{
  Plan plan;
  plan.plan_time = plan_time;
  plan.error = error;
  plan.valid = false;
  plan.fire_admissible = false;
  return plan;
}

[[nodiscard]] double centerYaw(const L3Estimation::TrackedTarget& target)
{
  const Eigen::VectorXd x = target.ekf_x();
  return std::atan2(x[2], x[0]);
}

}  // namespace

Planner::Planner(PlanConfig config)
: config_(std::move(config))
{
}

void Planner::reset() noexcept
{
  // SP Aimer 没有 reset；目标丢失、候选为空或弹道失败都不会清 lock_id_。
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

Planner::AimPoint Planner::chooseAimPoint(
  const L3Estimation::TrackedTarget& target)
{
  const Eigen::VectorXd ekf_x = target.ekf_x();
  const std::vector<Eigen::Vector4d> armors = target.armor_xyza_list();
  if (armors.empty() || ekf_x.size() < 11) {
    return {};
  }

  const double center_yaw = centerYaw(target);
  std::vector<double> delta_angles;
  delta_angles.reserve(armors.size());
  for (const auto& armor : armors) {
    delta_angles.push_back(
      L6Telemetry::limit_rad(armor[3] - center_yaw));
  }

  const auto pointAt = [&](int armor_id) {
    AimPoint point;
    point.valid = true;
    point.armor_id = armor_id;
    point.xyza = armors[static_cast<std::size_t>(armor_id)];
    point.delta_angle = delta_angles[static_cast<std::size_t>(armor_id)];
    return point;
  };

  // SP：没有发生过跳板时只信 0 号板，而且不触碰旧锁。
  if (!target.jumped) {
    return pointAt(0);
  }

  // 注意 SP 源码实际判断的是 x[8]（半径），不是角速度 x[7]。这里原样保留。
  if (std::abs(ekf_x[8]) <= 2.0 &&
      target.name != L3Estimation::ArmorName::Outpost) {
    std::vector<int> ids;
    ids.reserve(armors.size());
    for (std::size_t id = 0; id < armors.size(); ++id) {
      if (std::abs(delta_angles[id]) > 60.0 / 57.3) {
        continue;
      }
      ids.push_back(static_cast<int>(id));
    }

    // SP 直接返回 invalid，且不清锁。
    if (ids.empty()) {
      return {};
    }

    // SP 只比较窗口内的前两块；平局由严格 < 落到第二块。
    if (ids.size() > 1) {
      const int id0 = ids[0];
      const int id1 = ids[1];
      if (locked_id_ != id0 && locked_id_ != id1) {
        locked_id_ =
          std::abs(delta_angles[static_cast<std::size_t>(id0)]) <
              std::abs(delta_angles[static_cast<std::size_t>(id1)])
            ? id0
            : id1;
      }
      return pointAt(locked_id_);
    }

    // 只有一块进入窗口时，SP 才退出锁定模式。
    locked_id_ = -1;
    return pointAt(ids[0]);
  }

  double coming_angle = config_.selector.coming_angle;
  double leaving_angle = config_.selector.leaving_angle;
  if (target.name == L3Estimation::ArmorName::Outpost) {
    coming_angle = 70.0 / 57.3;
    leaving_angle = 30.0 / 57.3;
  }

  for (std::size_t id = 0; id < armors.size(); ++id) {
    const double delta = delta_angles[id];
    if (std::abs(delta) > coming_angle) {
      continue;
    }
    if (ekf_x[7] > 0.0 && delta < leaving_angle) {
      return pointAt(static_cast<int>(id));
    }
    if (ekf_x[7] < 0.0 && delta > -leaving_angle) {
      return pointAt(static_cast<int>(id));
    }
  }

  return {};
}

Plan Planner::plan(const PlanInput& input)
{
  if (!input.target.has_value()) {
    return rejected(PlanError::NoTarget, input.plan_time);
  }
  if (input.target->armor_num() < 1 || input.target->ekf_x().size() < 11) {
    return rejected(PlanError::NoArmor, input.plan_time);
  }

  L3Estimation::TrackedTarget target = *input.target;
  const Eigen::VectorXd target_x = target.ekf_x();

  // SP 使用有符号比较；负向高速旋转仍走 low delay。
  const double delay_time = target_x[7] > config_.decision_speed
    ? config_.high_speed_delay_time
    : config_.low_speed_delay_time;

  double bullet_speed = input.robot_state.bullet_speed;
  const bool bullet_speed_ok = config_.bulletSpeedValid(bullet_speed);
  if (!bullet_speed_ok) {
    bullet_speed = config_.fallback_bullet_speed;
  }

  Delay delay;
  delay.image_to_plan = input.to_now
    ? std::chrono::duration<double>(input.plan_time - target.t()).count()
    : 0.005;
  delay.control_to_fire = delay_time;
  const double before_fire = delay.beforeFire();

  const TimePoint future = target.t() + spDuration(before_fire);
  target.predict(future);

  AimPoint final_aim = chooseAimPoint(target);
  if (!final_aim.valid) {
    return rejected(PlanError::OutOfWindow, input.plan_time);
  }

  const Eigen::Vector3d xyz0 = final_aim.xyza.head<3>();
  const double distance0 = std::hypot(xyz0.x(), xyz0.y());
  SpTrajectory current_trajectory =
    solveSpTrajectory(bullet_speed, distance0, xyz0.z());
  if (current_trajectory.unsolvable) {
    return rejected(PlanError::BallisticFailed, input.plan_time);
  }

  double previous_fly_time = current_trajectory.fly_time;
  const double tolerance =
    std::chrono::duration<double>(config_.fly_time_tolerance).count();

  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    // 每轮都从同一个 prefire Target 副本预测到共同命中时刻；不能在上一轮
    // 结果上累计预测，也不能固定板号各自迭代。
    L3Estimation::TrackedTarget iteration_target = target;
    const TimePoint predict_time = future + spDuration(previous_fly_time);
    iteration_target.predict(predict_time);

    final_aim = chooseAimPoint(iteration_target);
    if (!final_aim.valid) {
      return rejected(PlanError::OutOfWindow, input.plan_time);
    }

    const Eigen::Vector3d xyz = final_aim.xyza.head<3>();
    const double distance = std::hypot(xyz.x(), xyz.y());
    current_trajectory =
      solveSpTrajectory(bullet_speed, distance, xyz.z());
    if (current_trajectory.unsolvable) {
      return rejected(PlanError::BallisticFailed, input.plan_time);
    }

    if (std::abs(current_trajectory.fly_time - previous_fly_time) < tolerance) {
      break;
    }
    previous_fly_time = current_trajectory.fly_time;
  }

  Plan plan;
  plan.target_id = static_cast<int>(target.name);
  plan.armor_id = final_aim.armor_id;
  plan.plan_time = input.plan_time;
  plan.fire_time = future;
  plan.hit_time = future + spDuration(current_trajectory.fly_time);
  plan.aim_point = final_aim.xyza.head<3>();
  plan.yaw =
    std::atan2(plan.aim_point.y(), plan.aim_point.x()) + config_.yaw_offset;
  plan.pitch = -(current_trajectory.pitch + config_.pitch_offset);
  plan.fly_time = current_trajectory.fly_time;
  delay.fire_to_hit = current_trajectory.fly_time;
  plan.delay = delay;
  plan.ballistic_valid = true;
  plan.aim_phase = AimPhase::SingleArmor;
  plan.aim_on_armor = true;
  plan.fire_armor_id = final_aim.armor_id;
  plan.fire_delta_angle = final_aim.delta_angle;
  plan.fire_armor_point = final_aim.xyza.head<3>();
  plan.fire_admissible = bullet_speed_ok;
  plan.type = PlanType::Setpoint;
  plan.error = bullet_speed_ok ? PlanError::None : PlanError::BadBulletSpeed;
  plan.valid = true;

  if (!std::isfinite(plan.yaw) || !std::isfinite(plan.pitch) ||
      !std::isfinite(plan.fly_time)) {
    return rejected(PlanError::BallisticFailed, input.plan_time);
  }
  return plan;
}

}  // namespace L4Planning
