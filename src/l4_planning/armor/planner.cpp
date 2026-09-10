#include "l4_planning/armor/planner.hpp"

#include "l6_telemetry/math.hpp"

#include <chrono>
#include <cmath>
#include <limits>
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
: config_(config), ballistic_(config.ballistic), smoother_(config.blend.limits)
{
}

Plan Planner::plan(const PlanInput& input)
{
  if (!input.target.has_value()) {
    // 目标没了，正在进行的过渡段所依据的切板预测随之失效，不能继续按它走。
    smoother_.reset();
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
    if (auto blended = blendOnlyPlan(input.plan_time, delay)) {
      return *blended;
    }
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
      if (auto blended = blendOnlyPlan(input.plan_time, delay)) {
        return *blended;
      }
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
  const double shoot_yaw =
    std::atan2(point.y(), point.x()) + config_.impact.yaw_offset;
  // 世界系约定 pitch 向下为正，因此弹道仰角在此取反。
  const double shoot_pitch =
    -(current_trajectory.pitch + config_.impact.pitch_offset);
  last_shoot_yaw_ = shoot_yaw;
  last_shoot_pitch_ = shoot_pitch;
  has_last_shoot_ = true;

  plan.aim.point = point;
  plan.aim.shoot_yaw = shoot_yaw;
  plan.aim.shoot_pitch = shoot_pitch;
  // 默认下发射击轨迹原值；开了过渡段才可能被多项式改写。
  plan.aim.yaw = shoot_yaw;
  plan.aim.pitch = shoot_pitch;

  if (config_.blend.enable) {
    const auto samplerFor = [this, &target, fly_time = current_trajectory.fly_time,
                             bullet_speed](int armor_id) {
      return [this, target, fly_time, bullet_speed, armor_id](double offset) {
        return sampleTrajectory(target, fly_time, bullet_speed, offset, armor_id);
      };
    };

    std::optional<AimSmoother::Forecast> forecast;
    int next_id = -1;
    if (smoother_.blending() && blend_target_id_ >= 0) {
      // 过渡进行中：目标板在提交那一刻就定死了，不能跟着 final_aim 漂。切板
      // 一旦真的发生，final_aim.armor_id 就成了新板，nextSwitchTime 给出的是
      // 再下一块，基准轨迹会整块跳到别处去。switch_time 与 before 此时都用不
      // 上，AimSmoother 在 active 分支里只取 after。
      AimSmoother::Forecast candidate;
      // switch_time 置非有限：这份预报只负责续供基准轨迹，不能被当成一次新的
      // 切板预测。过渡走完的那一帧 update() 会先 reset 再看提交条件，
      // 拿一个 switch_time = 0 的续供去判，会每帧都重新提交，过渡永远结束不了。
      candidate.switch_time = std::numeric_limits<double>::quiet_NaN();
      candidate.after = samplerFor(blend_target_id_);
      // 射击轨迹这一帧是不是已经就是目标板了。是的话过渡可以安全交还，
      // 不是的话 AimSmoother 会停在目标板轨迹上继续等。
      candidate.destination_selected = final_aim.armor_id == blend_target_id_;
      forecast = std::move(candidate);
    } else if (const auto switch_time = nextSwitchTime(
                 target, current_trajectory.fly_time, final_aim.armor_id,
                 next_id)) {
      AimSmoother::Forecast candidate;
      candidate.switch_time = *switch_time;
      candidate.before = sampleTrajectory(
        target, current_trajectory.fly_time, bullet_speed, 0.0,
        final_aim.armor_id);
      // 终点钉在切板后那一块上，不重新选板：过渡的意义就是"提前奔向下一块"。
      candidate.after = samplerFor(next_id);
      forecast = std::move(candidate);
    }

    const auto smoothed =
      smoother_.update(input.plan_time, shoot_yaw, shoot_pitch, forecast);
    // 记住这一段过渡奔向的是哪一块板，后续帧据此重采基准轨迹。
    if (!smoothed.blending) {
      blend_target_id_ = -1;
    } else if (blend_target_id_ < 0) {
      blend_target_id_ = next_id;
    }
    plan.aim.yaw = smoothed.yaw;
    plan.aim.pitch = smoothed.pitch;
    plan.aim.blending = smoothed.blending;
    plan.blend = BlendStatus{
      smoothed.peak_yaw_acceleration,
      smoothed.peak_pitch_acceleration,
      smoothed.acceleration_limited,
      smoothed.late_by};
  }

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
  // 过渡段则必须清：它锁死的系数来自一次已经作废的切板预测。
  smoother_.reset();
  blend_target_id_ = -1;
  has_last_shoot_ = false;
}

// ---- 以下为私有实现 ----

AimState Planner::sampleTrajectory(
  const L3Estimation::TrackedTarget& target_at_fire,
  double fly_time,
  double bullet_speed,
  double offset,
  int armor_id) const
{
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  const AxisState invalid_axis{kNaN, kNaN, kNaN};
  const AimState invalid{invalid_axis, invalid_axis};

  const double step = config_.blend.derivative_step;
  if (!(step > 0.0) || !std::isfinite(offset)) {
    return invalid;
  }

  // 三点中心差分：[offset - step, offset, offset + step]。
  double yaw[3];
  double pitch[3];
  for (int i = 0; i < 3; ++i) {
    L3Estimation::TrackedTarget probe = target_at_fire;
    probe.predict(fly_time + offset + (i - 1) * step);

    const std::vector<Eigen::Vector4d> armors = probe.armor_xyza_list();
    if (armor_id < 0 ||
        static_cast<std::size_t>(armor_id) >= armors.size()) {
      return invalid;
    }

    const Eigen::Vector3d position = armors[static_cast<std::size_t>(armor_id)].head<3>();
    const double distance = std::hypot(position.x(), position.y());
    const Ballistic trajectory =
      ballistic_.solve(distance, position.z(), bullet_speed);
    if (!trajectory.valid) {
      return invalid;
    }

    yaw[i] = std::atan2(position.y(), position.x()) + config_.impact.yaw_offset;
    pitch[i] = -(trajectory.pitch + config_.impact.pitch_offset);
  }

  // 差分一律走归一化差。yaw 出自 atan2，天然落在 (-pi, pi]，跨 ±pi 时直接
  // 相减会得到一个 2pi/step 的假尖峰，被当成"需要无穷大加速度"，过渡段就
  // 会在那一帧被判成不可行。
  const auto axis = [step](const double* value) {
    AxisState state;
    state.position = value[1];
    state.velocity = L6Telemetry::limit_rad(value[2] - value[0]) / (2.0 * step);
    state.acceleration =
      (L6Telemetry::limit_rad(value[2] - value[1]) -
       L6Telemetry::limit_rad(value[1] - value[0])) /
      (step * step);
    return state;
  };

  AimState state;
  state.yaw = axis(yaw);
  state.pitch = axis(pitch);
  return state;
}

std::optional<double> Planner::nextSwitchTime(
  const L3Estimation::TrackedTarget& target_at_fire,
  double fly_time,
  int current_id,
  int& next_id) const
{
  const double grid = config_.blend.grid;
  const double horizon = config_.blend.horizon;
  if (!(grid > 0.0) || !(horizon > grid)) {
    return std::nullopt;
  }
  const int steps = static_cast<int>(horizon / grid);

  // 迟滞锁沿扫描逐步推进，模拟它逐帧真实的演化；但它是局部副本，前视绝不
  // 写回 locked_id_——否则"看一眼未来"就把当前帧的锁改掉了。
  int lock = locked_id_;
  for (int step = 1; step <= steps; ++step) {
    const double offset = grid * step;
    L3Estimation::TrackedTarget probe = target_at_fire;
    probe.predict(fly_time + offset);

    const AimPoint point = chooseAimPoint(probe, lock);
    // 空窗不算切板。前哨站两块板之间就有这么一段（进入角 70 度、离开角
    // 30 度，三板 120 度间隔，中间约 20 度谁都不满足），过渡段应当盖过它。
    if (!point.valid) {
      continue;
    }
    if (point.armor_id != current_id) {
      next_id = point.armor_id;
      return offset;
    }
  }
  return std::nullopt;
}

std::optional<Plan> Planner::blendOnlyPlan(TimePoint now, const Delay& delay)
{
  if (!config_.blend.enable || !smoother_.blending() || !has_last_shoot_) {
    return std::nullopt;
  }

  // 过渡段已锁死系数，这里只是求值；传上一次的射击轨迹原值只是为了在过渡
  // 恰好结束的那一帧有个合理的回落值。
  const auto smoothed =
    smoother_.update(now, last_shoot_yaw_, last_shoot_pitch_, std::nullopt);
  if (!smoothed.blending) {
    blend_target_id_ = -1;
    return std::nullopt;
  }

  Plan plan;
  // 没有实体装甲板可判，所以只能 TrackOnly：云台继续沿过渡段走，L5 会因为
  // plan.fire 为空而拒绝开火。原因仍记 OutOfWindow——本帧确实没有可击打板。
  plan.status = PlanStatus::TrackOnly;
  plan.reason = PlanError::OutOfWindow;
  plan.aim.yaw = smoothed.yaw;
  plan.aim.pitch = smoothed.pitch;
  plan.aim.shoot_yaw = last_shoot_yaw_;
  plan.aim.shoot_pitch = last_shoot_pitch_;
  plan.aim.blending = true;
  plan.blend = BlendStatus{
    smoothed.peak_yaw_acceleration,
    smoothed.peak_pitch_acceleration,
    smoothed.acceleration_limited,
    smoothed.late_by};
  plan.timing.delay = delay;
  return plan;
}

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

    // 两块板同时可见时锁定其中一块，后续帧沿用锁定结果，避免角度接近时来回切换。
    if (ids.size() > 1) {
      const int id0 = ids[0];
      const int id1 = ids[1];
      const auto facing = [&](int id) {
        return std::abs(delta_angles[static_cast<std::size_t>(id)]);
      };
      const int challenger = facing(id0) < facing(id1) ? id0 : id1;

      // 锁不在候选里（刚跟上目标，或在任板已经转出窗口）时无从沿用，取更正的那块。
      // 锁还在候选里就无条件保持：在任板是否该让位，只由它自己转出窗口来回答，
      // 不由"谁更正对"这种可以来回翻转的比较来回答。
      if (lock != id0 && lock != id1) {
        lock = challenger;
      }
      return pointAt(lock);
    }

    // 只剩一块候选：它就是新的在任板，锁要写成它而不是清空。清空的话下一帧
    // 第二块板进窗口时锁是空的，会走上面"无从沿用"的分支立刻改选更正的那块；
    // 那块一旦被噪声挤出窗口就又切回来，这正是实测里"换完板又甩回去"的来源。
    // 写成 ids[0] 之后，切板只由"在任板离开窗口"这个单调事件触发。
    lock = ids[0];
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
