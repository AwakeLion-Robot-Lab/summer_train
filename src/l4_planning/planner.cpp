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
  plan.fire_admissible = false;
  return plan;
}

// 整车中心相对枪口的方位角。装甲板法线与它的夹角为 0 表示正对枪口。
[[nodiscard]] double centerYaw(const L3Estimation::TargetState& target)
{
  return std::atan2(target.position.y(), target.position.x());
}

}  // namespace

Planner::Planner(PlanConfig config)
: config_(std::move(config)),
  ballistic_(config_.ballistic),
  phase_(config_.aim_phase)
{
}

void Planner::reset() noexcept
{
  locked_id_ = -1;
  phase_.reset();
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

Planner::Candidate Planner::refineArmor(
  const L3Estimation::TargetState& target, int armor_id, double before_fire,
  double bullet_speed) const
{
  Candidate candidate;
  candidate.armor_id = armor_id;

  const double tolerance =
    std::chrono::duration<double>(config_.fly_time_tolerance).count();

  // 不动点方程：命中时刻取决于飞行时间，飞行时间又取决于命中时刻的位置。
  // 板号在整个迭代里固定，所以每一步的目标是同一块板，必然收敛。
  double fly_time = 0.0;
  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const double total = before_fire + fly_time;
    const auto predicted = predictor_.predict(target, total);
    const auto armors = predictor_.armorPoses(predicted);
    if (armor_id >= static_cast<int>(armors.size())) {
      return candidate;
    }

    const auto& xyza = armors[static_cast<std::size_t>(armor_id)];
    // world 系原点即枪管原点，水平距离和高度可直接取。
    const double distance = std::hypot(xyza.x(), xyza.y());
    const Ballistic ballistic = ballistic_.solve(distance, xyza.z(), bullet_speed);
    if (!ballistic.valid) {
      return candidate;
    }

    const bool converged = std::abs(ballistic.fly_time - fly_time) < tolerance;
    fly_time = ballistic.fly_time;

    if (converged) {
      candidate.valid = true;
      candidate.total_time = before_fire + fly_time;
      candidate.fly_time = fly_time;
      candidate.xyza = xyza;
      candidate.delta_angle =
        L6Telemetry::limit_rad(xyza[3] - centerYaw(predicted));
      return candidate;
    }
  }

  return candidate;
}

int Planner::selectArmor(
  const std::vector<Candidate>& candidates, bool geometry_observed, bool& degraded)
{
  degraded = false;

  // 整车几何还没被观测约束过：除了当前正在观测的 0 号板，其余板的位置完全
  // 由初值猜出来。此时瞄别的板等于拿伪造的几何开火。
  if (!geometry_observed) {
    locked_id_ = -1;
    return candidates[0].valid ? 0 : -1;
  }

  int best = -1;
  double best_angle = std::numeric_limits<double>::infinity();
  int fallback = -1;
  double fallback_angle = std::numeric_limits<double>::infinity();

  for (const auto& candidate : candidates) {
    if (!candidate.valid) continue;
    const double angle = std::abs(candidate.delta_angle);

    if (angle < fallback_angle) {
      fallback_angle = angle;
      fallback = candidate.armor_id;
    }
    if (angle > config_.selector.front_window) continue;
    if (angle < best_angle) {
      best_angle = angle;
      best = candidate.armor_id;
    }
  }

  if (best < 0) {
    // 前置窗口里一块板都没有。仍然给出角度让云台继续跟随，只是标记降级由
    // 火控拒绝——停止跟随比指偏更糟，等窗口回来时已经找不着目标了。
    degraded = true;
    locked_id_ = -1;
    return fallback;
  }

  // 锁定迟滞：两块板都接近窗口边缘时会逐帧互换，命令抖动到云台跟不上。
  // 已锁定的板要差过迟滞门限才允许换。
  if (locked_id_ >= 0 && locked_id_ < static_cast<int>(candidates.size())) {
    const auto& locked = candidates[static_cast<std::size_t>(locked_id_)];
    const double locked_angle = std::abs(locked.delta_angle);
    if (locked.valid && locked_angle <= config_.selector.front_window &&
        locked_angle - best_angle < config_.selector.switch_hysteresis) {
      best = locked_id_;
    }
  }

  locked_id_ = best;
  return best;
}

Eigen::Vector3d Planner::projectCenterAim(
  const L3Estimation::TargetState& predicted, double radius, double armor_z)
{
  // 枪口在 world 系原点，所以指向中心的射线就是中心的水平方向单位向量。
  Eigen::Vector2d ray = predicted.position.head<2>();
  const double norm = ray.norm();
  if (norm > 1e-6) {
    ray /= norm;
  } else {
    ray = Eigen::Vector2d{1.0, 0.0};
  }

  // 沿射线退回一个半径，就是旋转圆上离枪口最近的点——装甲板扫过这里时正对
  // 枪口。高度取实体板的高度，不是中心高度。
  const Eigen::Vector2d projected = predicted.position.head<2>() - radius * ray;
  return Eigen::Vector3d{projected.x(), projected.y(), armor_z};
}

bool Planner::inFireWindow(
  const L3Estimation::TargetState& target, double delta_angle) const
{
  const auto& selector = config_.selector;
  const bool is_outpost = target.name == L3Estimation::ArmorName::Outpost;
  const double coming =
    is_outpost ? selector.outpost_coming_angle : selector.coming_angle;
  const double leaving =
    is_outpost ? selector.outpost_leaving_angle : selector.leaving_angle;

  if (std::abs(delta_angle) > coming) {
    return false;
  }

  // 转速低到几乎不转时无所谓转入转出，只看正不正对。
  if (std::abs(target.v_yaw) < 1e-3) {
    return true;
  }

  // v_yaw > 0 时 delta_angle 递增，尚未越过 leaving 线的板才是正在转入的那
  // 一块；反向旋转时判据镜像。转出侧的板等子弹飞到时已经背对枪口。
  return target.v_yaw > 0.0 ? delta_angle < leaving : delta_angle > -leaving;
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
  if (target->armor_num < 1) {
    reset();
    return rejected(PlanError::NoArmor, plan_time);
  }

  // 弹速为 0 是裁判系统上电初期的正常值。这里用兜底初速保证仍能解算并输出
  // 瞄准角，但把 BadBulletSpeed 记进 error，由 L5 拒绝开火。
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
  const double before_fire = delay.beforeFire();

  // 档位由整车角速度驱动，几何不可观测时强制留在 SingleArmor。
  phase_.update(std::abs(target->v_yaw), target->multi_armor_observed);
  const AimPhase phase = phase_.phase();

  // 逐板求解不动点，再在收敛后的候选里选板。
  std::vector<Candidate> candidates;
  candidates.reserve(static_cast<std::size_t>(target->armor_num));
  for (int id = 0; id < target->armor_num; ++id) {
    candidates.push_back(refineArmor(*target, id, before_fire, bullet_speed));
  }

  bool degraded = false;
  const int armor_id =
    selectArmor(candidates, target->multi_armor_observed, degraded);
  if (armor_id < 0) {
    // 每块板都没能收敛出弹道解，说明目标在射程外或参数非法。
    return rejected(PlanError::BallisticFailed, plan_time);
  }

  const Candidate& chosen = candidates[static_cast<std::size_t>(armor_id)];
  const auto predicted = predictor_.predict(*target, chosen.total_time);

  // WholeCarCenter 档瞄旋转圆上离枪口最近的点，不瞄某块具体的板。
  const bool aim_on_armor = phase != AimPhase::WholeCarCenter;
  const bool use_alternate =
    target->armor_num == 4 && (armor_id == 1 || armor_id == 3);
  const double radius = use_alternate ? target->second_radius : target->radius;
  const Eigen::Vector3d aim_point =
    aim_on_armor ? Eigen::Vector3d{chosen.xyza.head<3>()}
                 : projectCenterAim(predicted, radius, chosen.xyza.z());

  // 代理点与实体板的距离不同，弹道要按实际瞄准点重解一次。
  const double distance = std::hypot(aim_point.x(), aim_point.y());
  const Ballistic ballistic =
    ballistic_.solve(distance, aim_point.z(), bullet_speed);
  if (!ballistic.valid) {
    return rejected(PlanError::BallisticFailed, plan_time);
  }

  Plan plan;
  plan.target_id = target->target_id;
  plan.armor_id = armor_id;
  plan.plan_time = plan_time;
  plan.fire_time = plan_time + toDuration(before_fire - delay.image_to_plan);
  plan.hit_time = plan.fire_time + toDuration(ballistic.fly_time);
  plan.aim_point = aim_point;

  // yaw 由水平分量直接求得；pitch 抬头为正，下位机的符号约定由 L5 施加。
  plan.yaw = std::atan2(aim_point.y(), aim_point.x());
  plan.pitch = ballistic.pitch;
  plan.fly_time = ballistic.fly_time;
  delay.fire_to_hit = ballistic.fly_time;
  plan.delay = delay;
  plan.ballistic_valid = true;
  plan.aim_phase = phase;
  plan.aim_on_armor = aim_on_armor;
  plan.type = PlanType::Setpoint;
  plan.valid = true;

  if (!std::isfinite(plan.yaw) || !std::isfinite(plan.pitch)) {
    return rejected(PlanError::BallisticFailed, plan_time);
  }

  // 火控永远针对**实体装甲板**判定，即使这一帧瞄的是中心代理点——否则中心
  // 档会在两块板之间的空档里照样开火。选正对枪口的那块作为火控依据。
  int fire_id = -1;
  double fire_angle = std::numeric_limits<double>::infinity();
  for (const auto& candidate : candidates) {
    if (!candidate.valid) continue;
    if (std::abs(candidate.delta_angle) < std::abs(fire_angle)) {
      fire_angle = candidate.delta_angle;
      fire_id = candidate.armor_id;
    }
  }
  plan.fire_armor_id = fire_id;
  plan.fire_delta_angle = fire_id >= 0 ? fire_angle : 0.0;

  const bool in_window = fire_id >= 0 && inFireWindow(*target, fire_angle);
  plan.fire_admissible = bullet_speed_ok && !degraded && in_window;

  // 单个 error 只能带一个降级原因，弹速异常比窗口未命中更根本，优先报它。
  if (!bullet_speed_ok) {
    plan.error = PlanError::BadBulletSpeed;
  } else if (!plan.fire_admissible) {
    plan.error = PlanError::OutOfWindow;
  } else {
    plan.error = PlanError::None;
  }
  return plan;
}

}  // namespace L4Planning
