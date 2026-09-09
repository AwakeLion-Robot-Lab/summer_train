#include "l4_planning/armor/planner.hpp"

#include "l6_telemetry/math.hpp"

#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

namespace L4Planning {
namespace {

// 规划时间统一量化到微秒，保证每轮迭代使用相同的时间精度。
std::chrono::microseconds secondsToDuration(double seconds)
{
  return std::chrono::microseconds(static_cast<int>(seconds * 1e6));
}

Plan rejected(PlanError error)
{
  Plan plan;
  plan.reason = error;
  return plan;
}

double centerYaw(const L3Estimation::TrackedTarget& target)
{
  const Eigen::VectorXd x = target.ekf_x();
  return std::atan2(x[2], x[0]);
}

}  // namespace

Planner::Planner(ArmorPlanConfig config)
: config_(config), ballistic_(config.ballistic)
{
}

Plan Planner::plan(const PlanInput& input)
{
  if (!input.target.has_value()) {
    return rejected(PlanError::NoTarget);
  }

  L3Estimation::TrackedTarget target = *input.target;
  const Eigen::VectorXd target_x = target.ekf_x();

  // 当前延迟分档使用有符号 yaw 角速度：只有正向超过阈值才使用高速延迟。
  const double delay_time = target_x[7] > config_.impact.decision_speed
    ? config_.impact.high_speed_delay_time
    : config_.impact.low_speed_delay_time;

  double bullet_speed = input.robot_state.bullet_speed;
  const bool bullet_speed_measured =
    config_.impact.bulletSpeedValid(bullet_speed);
  if (!bullet_speed_measured) {
    // 弹速异常时用回退值生成跟随角。默认仍降级为 TrackOnly；只有显式打开
    // trust_fallback_bullet_speed 才把这个猜测值当实测值放行。
    bullet_speed = config_.impact.fallback_bullet_speed;
  }
  const bool bullet_speed_ok =
    bullet_speed_measured || config_.impact.trust_fallback_bullet_speed;

  Delay delay;
  // 实时运行直接测量曝光到规划的耗时；离线入口使用固定 5 ms。
  delay.image_to_plan = input.to_now
    ? std::chrono::duration<double>(input.plan_time - target.t()).count()
    : 0.005;
  // 规划到发送由 runtime 实测后回灌；串口到电控只能实车标定，未标定按 0 计，
  // 同时把计划降级成 TrackOnly，不允许在缺段的延迟上开火。
  delay.plan_to_send = input.plan_to_send;
  delay.send_to_control = config_.impact.send_to_control.value_or(0.0);
  delay.control_to_fire = delay_time;
  const double before_fire = delay.beforeFire();

  // 先把目标从图像时刻外推到预计发射时刻。
  const TimePoint future = target.t() + secondsToDuration(before_fire);
  target.predict(future);

  // 迟滞锁在整个迭代期间只读。迭代是在猜同一帧的落点，中间轮次的构型都是
  // 假想的，不该改变跨帧的迟滞状态；锁的新值先留在局部量里，收敛后一次性提交。
  // sp_vision 与 Climber 在这里都是让 choose_aim_point 直接写成员，awakening
  // 则把 select_armor 提到循环外只调一次（very_aimer.cpp:231 roughly_select，
  // 循环内那次调用被显式注释掉了）。这里取 awakening 的语义，但保留逐轮重选
  // ——落点变了可击打的板也会变，选板本身仍该跟着迭代走，只是不落锁。
  int lock = locked_id_;
  int probe_lock = lock;
  AimPoint final_aim = chooseAimPoint(target, probe_lock);
  if (!final_aim.valid) {
    return rejected(PlanError::OutOfWindow);
  }

  const Eigen::Vector3d xyz0 = final_aim.xyza.head<3>();
  const double distance0 = std::hypot(xyz0.x(), xyz0.y());
  Ballistic current_trajectory = ballistic_.solve(distance0, xyz0.z(), bullet_speed);
  if (!current_trajectory.valid) {
    return rejected(PlanError::BallisticFailed);
  }

  int converged_lock = lock;
  double previous_fly_time = current_trajectory.fly_time;
  const double tolerance =
    std::chrono::duration<double>(config_.impact.fly_time_tolerance).count();

  for (int iteration = 0; iteration < config_.impact.max_iterations; ++iteration) {
    // 飞行时间决定命中时刻，命中点又会改变飞行时间。每轮都从同一个发射时刻
    // 状态重新外推，避免把上一轮的 dt 重复累计。
    L3Estimation::TrackedTarget iteration_target = target;
    const TimePoint predict_time = future + secondsToDuration(previous_fly_time);
    iteration_target.predict(predict_time);

    int iteration_lock = lock;
    final_aim = chooseAimPoint(iteration_target, iteration_lock);
    if (!final_aim.valid) {
      return rejected(PlanError::OutOfWindow);
    }

    const Eigen::Vector3d xyz = final_aim.xyza.head<3>();
    const double distance = std::hypot(xyz.x(), xyz.y());
    current_trajectory = ballistic_.solve(distance, xyz.z(), bullet_speed);
    if (!current_trajectory.valid) {
      return rejected(PlanError::BallisticFailed);
    }

    converged_lock = iteration_lock;
    if (std::abs(current_trajectory.fly_time - previous_fly_time) < tolerance) {
      break;
    }
    previous_fly_time = current_trajectory.fly_time;
  }

  // 只有走到这里才说明本帧真的解出了瞄准点；中途 return 的分支一律不动锁，
  // 与"短暂中断不清锁"的既有语义一致。
  locked_id_ = converged_lock;

  const Eigen::Vector3d point = final_aim.xyza.head<3>();
  delay.fire_to_hit = current_trajectory.fly_time;

  Plan plan;
  // 两种降级：回退弹速生成的解不能标成可开火；延迟链缺实车标定段时落点会
  // 系统性偏早，同样只跟随。弹速优先报，因为它同时也让弹道解本身失真。
  if (!bullet_speed_ok) {
    plan.status = PlanStatus::TrackOnly;
    plan.reason = PlanError::BadBulletSpeed;
  } else if (!config_.impact.fireDelayReady()) {
    plan.status = PlanStatus::TrackOnly;
    plan.reason = PlanError::DelayNotCalibrated;
  } else {
    plan.status = PlanStatus::FireReady;
    plan.reason = PlanError::None;
  }
  plan.aim = AimReference{
    point,
    std::atan2(point.y(), point.x()) + config_.impact.yaw_offset,
    // 世界系约定 pitch 向下为正，因此弹道仰角在此取反。
    -(current_trajectory.pitch + config_.impact.pitch_offset)};
  plan.fire = FireReference{final_aim.armor_id, final_aim.xyza};
  plan.timing = PlanTiming{
    future + secondsToDuration(current_trajectory.fly_time),
    current_trajectory.fly_time,
    delay};

  if (!std::isfinite(plan.aim.yaw) || !std::isfinite(plan.aim.pitch) ||
      !std::isfinite(plan.timing.fly_time)) {
    return rejected(PlanError::BallisticFailed);
  }
  return plan;
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

void Planner::reset() noexcept
{
  // locked_id_ 由候选板变化时更新。短暂中断不清锁，避免恢复后立即切板。
}

// ---- 以下为私有实现 ----

Planner::AimPoint Planner::chooseAimPoint(
  const L3Estimation::TrackedTarget& target, int& lock) const
{
  const Eigen::VectorXd ekf_x = target.ekf_x();
  const std::vector<Eigen::Vector4d> armors = target.armor_xyza_list();
  if (armors.empty()) {
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
    return point;
  };

  // 尚未发生过板间关联跳变时，L3 只确认了当前观测板（0 号板）。
  if (!target.jumped) {
    return pointAt(0);
  }

  // x[8] 是第一组装甲板半径；正常尺寸车辆通常进入这一常规选板分支。
  if (std::abs(ekf_x[8]) <= 2.0 &&
      target.name != L3Estimation::ArmorName::Outpost) {
    std::vector<int> ids;
    ids.reserve(armors.size());
    for (std::size_t id = 0; id < armors.size(); ++id) {
      if (std::abs(delta_angles[id]) > config_.selector.coming_angle) {
        continue;
      }
      ids.push_back(static_cast<int>(id));
    }

    // 当前没有正面候选板，返回无有效瞄准点；旧锁保留到候选重新出现。
    if (ids.empty()) {
      return {};
    }

    // 两块板同时可见时锁定其中朝向更正的一块，后续帧沿用锁定结果，
    // 避免在角度接近时来回切换。
    if (ids.size() > 1) {
      const int id0 = ids[0];
      const int id1 = ids[1];
      if (lock != id0 && lock != id1) {
        lock =
          std::abs(delta_angles[static_cast<std::size_t>(id0)]) <
              std::abs(delta_angles[static_cast<std::size_t>(id1)])
            ? id0
            : id1;
      }
      return pointAt(lock);
    }

    // 只剩一块候选时无需迟滞，退出双板锁定。
    lock = -1;
    return pointAt(ids[0]);
  }

  double coming_angle = config_.selector.coming_angle;
  double leaving_angle = config_.selector.leaving_angle;
  if (target.name == L3Estimation::ArmorName::Outpost) {
    coming_angle = config_.selector.outpost_coming_angle;
    leaving_angle = config_.selector.outpost_leaving_angle;
  }

  // 旋转目标先用 coming_angle 限制正面区域，再结合旋转方向和
  // leaving_angle 排除即将离开可击打区域的板。
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

}  // namespace L4Planning
