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
  if (!config_.mpc.enable) {
    return;
  }
  // 两轴只有加速度上限不同，其余参数共用一套。
  AxisMpc::Config axis;
  axis.dt = config_.mpc.dt;
  axis.horizon = config_.mpc.horizon;
  axis.q_position = config_.mpc.q_position;
  axis.q_velocity = config_.mpc.q_velocity;
  axis.r_input = config_.mpc.r_input;
  axis.rho = config_.mpc.rho;
  axis.max_iterations = config_.mpc.max_iterations;
  axis.tolerance = config_.mpc.tolerance;

  axis.max_acceleration = config_.mpc.max_yaw_acceleration;
  const bool yaw_ok = yaw_mpc_.setup(axis);
  axis.max_acceleration = config_.mpc.max_pitch_acceleration;
  const bool pitch_ok = pitch_mpc_.setup(axis);

  // 中心点要落在采样窗口内部，否则读不到中点；horizon 至少 3 步。
  mpc_ready_ = yaw_ok && pitch_ok && config_.mpc.horizon >= 3;
  if (mpc_ready_) {
    yaw_reference_.resize(2, config_.mpc.horizon);
    pitch_reference_.resize(2, config_.mpc.horizon);
  }
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
  // 射击角：打中命中点所需要的角，不受加速度约束整形。
  const double shoot_yaw =
    std::atan2(point.y(), point.x()) + config_.impact.yaw_offset;
  // 世界系约定 pitch 向下为正，因此弹道仰角在此取反。
  const double shoot_pitch =
    -(current_trajectory.pitch + config_.impact.pitch_offset);

  plan.aim = AimReference{};
  plan.aim.point = point;
  plan.aim.shoot_yaw = shoot_yaw;
  plan.aim.shoot_pitch = shoot_pitch;
  // 默认下发角就是射击角；下面整形成功才覆盖。
  plan.aim.yaw = shoot_yaw;
  plan.aim.pitch = shoot_pitch;

  double yaw_span = 0.0;
  double pitch_span = 0.0;
  if (mpc_ready_ &&
      buildReference(
        target, current_trajectory.fly_time, bullet_speed, shoot_yaw,
        shoot_pitch, yaw_reference_, pitch_reference_, yaw_span, pitch_span)) {
    const int center = config_.mpc.horizon / 2;
    const bool solved =
      yaw_mpc_.solve(yaw_reference_, yaw_reference_.col(0)) &&
      pitch_mpc_.solve(pitch_reference_, pitch_reference_.col(0));

    // 兜底闸门：整形是对参考的平滑，解不该跑出参考自身覆盖的角度范围。
    //
    // 不拿 ADMM 残差做这件事——实测残差在不同 rho 下不可比：rho=1/迭代10 的
    // 残差中位 0.05 却偏出 3.7 度，rho=10/迭代25 残差 2.15 反而只偏 0.025 度。
    // 而整形量本身有干净的界：50 条实车参考上，精确解的中点整形量中位 0.073
    // 度、最大 2.40 度，参考幅值中位 8.96 度，比值最大 0.296。所以拿"整形量不
    // 超过参考幅值"做闸门有三倍余量，又能挡住那种十几度的失控解。
    constexpr double kSpanFloor = 1.0e-3;  // 参考近乎不动时的数值噪声余量
    const bool sane = solved &&
      std::abs(yaw_mpc_.position(center)) <= yaw_span + kSpanFloor &&
      std::abs(pitch_mpc_.position(center)) <= pitch_span + kSpanFloor;
    if (sane) {
      // 参考存的是相对中心角的偏差，读出来要加回中心角。
      const double yaw_command =
        L6Telemetry::limit_rad(shoot_yaw + yaw_mpc_.position(center));
      const double pitch_command = shoot_pitch + pitch_mpc_.position(center);
      // 非有限值一律不采用，宁可这一帧不整形。
      if (std::isfinite(yaw_command) && std::isfinite(pitch_command)) {
        plan.aim.yaw = yaw_command;
        plan.aim.pitch = pitch_command;
        plan.aim.yaw_velocity = yaw_mpc_.velocity(center);
        plan.aim.yaw_acceleration = yaw_mpc_.acceleration(center);
        plan.aim.pitch_velocity = pitch_mpc_.velocity(center);
        plan.aim.pitch_acceleration = pitch_mpc_.acceleration(center);
        plan.aim.shaped = true;
      }
    }
  }

  plan.fire = FireReference{final_aim.armor_id, final_aim.xyza};
  plan.timing = PlanTiming{
    future + secondsToDuration(current_trajectory.fly_time),
    current_trajectory.fly_time,
    delay};

  if (!std::isfinite(plan.aim.shoot_yaw) || !std::isfinite(plan.aim.shoot_pitch) ||
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
  //
  // MPC 的热启动必须清：它跨帧复用上一次的解和对偶变量，10 次迭代才够用。
  // 目标丢了再回来时参考轨迹已经不连续，带着旧解迭代会把旧目标的偏置拖进来。
  yaw_mpc_.reset();
  pitch_mpc_.reset();
}

// ---- 以下为私有实现 ----

bool Planner::buildReference(
  const L3Estimation::TrackedTarget& target_at_fire,
  double fly_time,
  double bullet_speed,
  double center_yaw,
  double center_pitch,
  Eigen::Matrix<double, 2, Eigen::Dynamic>& yaw_reference,
  Eigen::Matrix<double, 2, Eigen::Dynamic>& pitch_reference,
  double& yaw_span,
  double& pitch_span) const
{
  const int steps = config_.mpc.horizon;
  const double dt = config_.mpc.dt;
  const int center = steps / 2;

  // 速度用中心差分，所以两端各要多采一个点：原始采样 steps + 2 个，
  // 原始下标 i 对应参考列 j = i - 1。
  const int raw_count = steps + 2;
  std::vector<double> raw_yaw(static_cast<std::size_t>(raw_count));
  std::vector<double> raw_pitch(static_cast<std::size_t>(raw_count));
  // 每个采样点选中的板号。速度只能在同一块板的采样点之间差分，见下面。
  std::vector<int> raw_id(static_cast<std::size_t>(raw_count));

  // 迟滞锁沿窗口正向推进，用局部副本，绝不写回成员——这是一次"看一眼整条
  // 轨迹"，不该改变当前帧的锁。
  //
  // 窗口起点在半个窗口之前，那时的锁没有记录，只能拿当前的锁做种子。小陀螺下
  // 这半秒车身已经转过大半圈，种子多半不在起点的候选里；chooseAimPoint 遇到
  // 这种情况会直接选最正对的一块，一两个采样点内就自愈，比为此保存一份历史
  // 命令轨迹划算得多。
  int lock = locked_id_;

  for (int i = 0; i < raw_count; ++i) {
    const double offset = static_cast<double>(i - 1 - center) * dt;

    L3Estimation::TrackedTarget probe = target_at_fire;
    // target_at_fire 是发射时刻的状态，再推一个飞行时间才是命中时刻。窗口就
    // 以这个命中时刻为中心：每个采样点回答"要打中那一刻的目标，枪得指哪"。
    probe.predict(fly_time + offset);

    const AimPoint aim = chooseAimPoint(probe, lock);
    if (!aim.valid) {
      return false;
    }
    const Eigen::Vector3d position = aim.xyza.head<3>();
    const Ballistic trajectory = ballistic_.solve(
      std::hypot(position.x(), position.y()), position.z(), bullet_speed);
    if (!trajectory.valid) {
      return false;
    }

    raw_yaw[static_cast<std::size_t>(i)] =
      std::atan2(position.y(), position.x()) + config_.impact.yaw_offset;
    raw_pitch[static_cast<std::size_t>(i)] =
      -(trajectory.pitch + config_.impact.pitch_offset);
    raw_id[static_cast<std::size_t>(i)] = aim.armor_id;
  }

  // 两轴都存成相对中心角的偏差：yaw 是为了避开 ±pi 跳变，pitch 是为了让参考
  // 末值落在 0 附近——终端代价用的是 Riccati 的 Pinf，参考绝对值越小它越不
  // 敏感。差分一律走归一化差，跨 ±pi 时直接相减会得到一个假尖峰。
  // 速度只在**同一块板**的采样点之间差分。跨切板做中心差分会把整板宽度的台阶
  // 除以 2*dt，得到一个几 rad/s 的假尖峰：3 m 处两块板的方位角差约 5.3 度，
  // dt=10 ms 时假速度就是 4.6 rad/s。它一旦落在第 0 列就成了 x0 的初速度，
  // MPC 带着它冲半个窗口，中点能偏出十几度——实测录像上每十几帧撞上一次，
  // 命令会出现单帧的大幅甩出再弹回。sp_vision 的 get_trajectory 是无条件中心
  // 差分，这里是一处有意的偏离。
  //
  // 位置上的台阶要保留：那才是切板本身，正是要交给 MPC 去平滑的东西。
  const auto axisVelocity = [&](const std::vector<double>& raw, std::size_t i,
                                bool wrap) {
    const auto difference = [wrap](double a, double b) {
      return wrap ? L6Telemetry::limit_rad(a - b) : a - b;
    };
    const bool same_before = raw_id[i - 1] == raw_id[i];
    const bool same_after = raw_id[i] == raw_id[i + 1];
    if (same_before && same_after) {
      return difference(raw[i + 1], raw[i - 1]) / (2.0 * dt);
    }
    if (same_before) {
      return difference(raw[i], raw[i - 1]) / dt;
    }
    if (same_after) {
      return difference(raw[i + 1], raw[i]) / dt;
    }
    // 前后都换了板：这一点孤立，给不出可信的速度，宁可报 0。
    return 0.0;
  };

  for (int j = 0; j < steps; ++j) {
    const std::size_t i = static_cast<std::size_t>(j + 1);
    yaw_reference(0, j) =
      L6Telemetry::limit_rad(raw_yaw[i] - center_yaw);
    yaw_reference(1, j) = axisVelocity(raw_yaw, i, true);
    pitch_reference(0, j) = raw_pitch[i] - center_pitch;
    pitch_reference(1, j) = axisVelocity(raw_pitch, i, false);
  }

  // 参考自身覆盖的角度范围。整形是对它的平滑，解不该跑出这个范围——用作下面
  // 的兜底闸门。
  yaw_span = yaw_reference.row(0).cwiseAbs().maxCoeff();
  pitch_span = pitch_reference.row(0).cwiseAbs().maxCoeff();

  return yaw_reference.allFinite() && pitch_reference.allFinite();
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
