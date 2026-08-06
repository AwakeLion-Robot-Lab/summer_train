#include "l4_planning/planner.hpp"

#include "l6_telemetry/math.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace L4Planning {
namespace {

// 把秒转成 steady_clock 的时长，用于填 Plan 里的三个时刻。
[[nodiscard]] TimePoint::duration toDuration(double seconds)
{
  return std::chrono::duration_cast<TimePoint::duration>(
    std::chrono::duration<double>(seconds));
}

[[nodiscard]] Plan rejected(PlanError error, TimePoint plan_time)
{
  Plan plan;
  plan.plan_time = plan_time;
  plan.error = error;
  plan.valid = false;
  return plan;
}

}  // namespace

Planner::Planner(PlanConfig config)
: config_(std::move(config)),
  ballistic_(config_.drag_coefficient)
{
}

void Planner::reset() noexcept
{
  locked_id_ = -1;
}

Plan Planner::plan(
  const std::optional<L3Estimation::TargetState>& target,
  const L1Sensor::RobotState& robot_state,
  TimePoint plan_time)
{
  PlanInput input;
  input.target = target;
  input.robot_state = robot_state;
  input.plan_time = plan_time;
  return plan(input);
}

Plan Planner::plan(const PlanInput& input)
{
  const auto& target = input.target;
  const auto& robot_state = input.robot_state;
  const TimePoint plan_time = input.plan_time;

  // 定点规划器不使用云台角速度；轨迹类规划器会在这里取边界条件。
  if (!target.has_value()) {
    reset();
    return rejected(PlanError::NoTarget, plan_time);
  }
  if (target->track_state == L3Estimation::TrackState::Lost) {
    reset();
    return rejected(PlanError::NotTracking, plan_time);
  }

  // 弹速为 0 是裁判系统上电初期的正常值。这里用兜底初速保证仍能解算并
  // 输出瞄准角，但把 BadBulletSpeed 记进 error，由 L5 拒绝开火。
  double bullet_speed = robot_state.bullet_speed;
  const bool bullet_speed_ok = std::isfinite(bullet_speed) &&
                               bullet_speed >= config_.min_valid_bullet_speed;
  if (!bullet_speed_ok) {
    bullet_speed = config_.fallback_bullet_speed;
  }

  // 曝光到规划是可测量段；标定段从配置取，未标定时为 0 并由
  // PlanConfig::fireDelayReady() 拦住开火。
  Delay delay;
  delay.image_to_plan = std::max(
    0.0, std::chrono::duration<double>(plan_time - target->timestamp).count());
  delay.send_to_control = config_.send_to_control.value_or(0.0);
  delay.control_to_fire = config_.control_to_fire.value_or(0.0);

  // 不动点迭代：命中时刻取决于飞行时间，飞行时间又取决于命中时刻的目标
  // 位置。以真空弹道的首解为起点，反复代入直到飞行时间稳定。
  const double tolerance =
    std::chrono::duration<double>(config_.fly_time_tolerance).count();
  double fly_time = 0.0;
  AimPoint aim;
  Ballistic ballistic;
  bool converged = false;

  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const double dt = delay.beforeFire() + fly_time;
    const auto predicted = predictor_.predict(*target, dt);
    const auto armors = predictor_.armorPoses(predicted);

    aim = chooseAimPoint(predicted, armors);
    if (!aim.valid) {
      return rejected(PlanError::NoArmor, plan_time);
    }

    // world 系原点即枪管原点，水平距离和高度可直接取。
    const double distance = std::hypot(aim.xyza.x(), aim.xyza.y());
    ballistic = ballistic_.solve(distance, aim.xyza.z(), bullet_speed);
    if (!ballistic.valid) {
      return rejected(PlanError::BallisticFailed, plan_time);
    }

    if (std::abs(ballistic.fly_time - fly_time) < tolerance) {
      fly_time = ballistic.fly_time;
      converged = true;
      break;
    }
    fly_time = ballistic.fly_time;
  }

  if (!converged) {
    // 迭代未收敛说明目标运动与弹道相互作用剧烈，此时的解不可信。
    return rejected(PlanError::BallisticFailed, plan_time);
  }

  delay.fire_to_hit = fly_time;

  Plan plan;
  plan.target_id = target->target_id;
  plan.armor_id = aim.armor_id;
  plan.plan_time = plan_time;
  plan.fire_time = plan_time + toDuration(delay.beforeFire() - delay.image_to_plan);
  plan.hit_time = plan.fire_time + toDuration(fly_time);
  plan.aim_point = aim.xyza.head<3>();

  // yaw 由水平分量直接求得；pitch 抬头为正，下位机的符号约定由 L5 施加。
  plan.yaw = std::atan2(aim.xyza.y(), aim.xyza.x());
  plan.pitch = ballistic.pitch;
  plan.fly_time = fly_time;
  plan.delay = delay;
  plan.ballistic_valid = true;
  plan.type = PlanType::Setpoint;
  plan.error = bullet_speed_ok ? PlanError::None : PlanError::BadBulletSpeed;
  plan.valid = true;

  if (!std::isfinite(plan.yaw) || !std::isfinite(plan.pitch)) {
    return rejected(PlanError::BallisticFailed, plan_time);
  }
  return plan;
}

Planner::AimPoint Planner::chooseAimPoint(
  const L3Estimation::TargetState& target,
  const std::vector<Eigen::Vector4d>& armors)
{
  AimPoint aim;
  if (armors.empty()) {
    reset();
    return aim;
  }

  // 整车中心相对枪口的方位角。delta_angle 为 0 表示该板正对枪口。
  const double center_yaw = std::atan2(target.position.y(), target.position.x());

  std::vector<double> delta_angles;
  delta_angles.reserve(armors.size());
  for (const auto& armor : armors) {
    delta_angles.push_back(L6Telemetry::limit_rad(armor[3] - center_yaw));
  }

  const auto& selector = config_.selector;
  const bool is_outpost = target.name == L3Estimation::ArmorName::Outpost;

  // 判据用角速度。sp_vision 的 aimer.cpp 这里写的是 ekf_x()[8]，那是半径
  // 而不是 v_yaw；半径恒在 0.05~0.5 之间、永远小于阈值 2，导致它的反陀螺
  // 分支对非前哨站目标是死代码。
  const bool spinning = std::abs(target.v_yaw) > selector.spin_threshold;

  if (!spinning && !is_outpost) {
    int best_id = -1;
    double best_angle = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index < armors.size(); ++index) {
      const double angle = std::abs(delta_angles[index]);
      if (angle > selector.max_face_angle) continue;
      if (angle < best_angle) {
        best_angle = angle;
        best_id = static_cast<int>(index);
      }
    }

    if (best_id < 0) {
      reset();
      return aim;
    }

    // 锁定迟滞：两块板都接近 45 度时会逐帧互换，命令抖动到云台跟不上。
    // 已锁定的板要差过迟滞门限才允许换。
    if (locked_id_ >= 0 && locked_id_ < static_cast<int>(armors.size())) {
      const double locked_angle = std::abs(delta_angles[locked_id_]);
      if (locked_angle <= selector.max_face_angle &&
          locked_angle - best_angle < selector.switch_hysteresis) {
        best_id = locked_id_;
      }
    }

    locked_id_ = best_id;
    aim.valid = true;
    aim.armor_id = best_id;
    aim.xyza = armors[best_id];
    aim.delta_angle = delta_angles[best_id];
    return aim;
  }

  // 反陀螺档：一侧的板不断转入视野，另一侧不断转出。只打转入的一侧，
  // 因为等子弹飞到时转出侧的板已经背对枪口。
  const double coming =
    is_outpost ? selector.outpost_coming_angle : selector.coming_angle;
  const double leaving =
    is_outpost ? selector.outpost_leaving_angle : selector.leaving_angle;

  for (std::size_t index = 0; index < armors.size(); ++index) {
    const double delta = delta_angles[index];
    if (std::abs(delta) > coming) continue;

    // v_yaw > 0 时 delta_angle 递增，尚未越过 leaving 线的板才是正在转入
    // 的那一块；反向旋转时判据镜像。
    const bool coming_in = target.v_yaw > 0.0 ? delta < leaving : delta > -leaving;
    if (!coming_in) continue;

    locked_id_ = static_cast<int>(index);
    aim.valid = true;
    aim.armor_id = locked_id_;
    aim.xyza = armors[index];
    aim.delta_angle = delta;
    return aim;
  }

  // 高速旋转时窗口内可能一块板都没有，属于正常间歇。
  reset();
  return aim;
}

}  // namespace L4Planning
