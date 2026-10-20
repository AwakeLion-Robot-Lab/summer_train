#include "l4_planning/buff/planner.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace L4Planning::Buff {
namespace {

constexpr std::array<double, 5> kBladeDegrees{0.0, 72.0, 144.0, 216.0, 288.0};

struct AttackGeometry {
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  std::array<Eigen::Vector3d, 4> box{};
};

AttackGeometry geometryAt(const L3Estimation::Buff::State& state, int blade)
{
  const double face = state.face_yaw;
  const double alpha = state.rotation_angle +
    kBladeDegrees[static_cast<std::size_t>(blade)] * std::numbers::pi / 180.0;
  const Eigen::Matrix3d rotation =
    Eigen::AngleAxisd{face, Eigen::Vector3d::UnitZ()}.toRotationMatrix();
  // radial 指向靶面相对符中心的径向，left 是同一符面内与其正交的方向。
  // alpha=0 时分别对应 +Z 和 +Y，与源 PagePoints 的 T/L/B/R 定义一致。
  const Eigen::Vector3d radial = rotation * Eigen::Vector3d{
    0.0, -std::sin(alpha), std::cos(alpha)};
  const Eigen::Vector3d left = rotation * Eigen::Vector3d{
    0.0, std::cos(alpha), std::sin(alpha)};

  AttackGeometry result;
  result.center = state.direction() + L3Estimation::Buff::kGlobalRadius * radial;
  const double half = L3Estimation::Buff::kBullseyeRadius;
  result.box = {
    result.center + half * (radial + left),
    result.center + half * (-radial + left),
    result.center + half * (-radial - left),
    result.center + half * (radial - left)};
  return result;
}

}  // namespace

Planner::Planner(Config config)
: config_(config), scheduler_(config.idle_duration, config.shoot_duration)
{
}

void Planner::reset() noexcept
{
  scheduler_.reset();
}

Plan Planner::plan(const L3Estimation::Buff::State& state,
                   TimePoint plan_time,
                   BulletSpeed bullet_speed,
                   bool to_now)
{
  Plan result;
  result.center = state.direction();
  result.attack = result.center;
  result.bullet_speed = bullet_speed;

  const double current_age = std::chrono::duration<double>(
    plan_time - state.timestamp).count();
  const double base_delay = config_.shoot_delay + (to_now ? current_age : 0.0);
  const double initial_fly = result.center.norm() / bullet_speed.value;
  if (!std::isfinite(initial_fly) || bullet_speed.value <= 0.0) {
    result.degraded = true;
    result.pre_aim = !scheduler_.started();
    return result;
  }

  double fly_time = initial_fly;
  int blade = -1;
  Eigen::Vector3d attack = result.center;
  std::array<Eigen::Vector3d, 4> attack_box{};
  bool attack_box_valid = false;
  const auto convergence = std::chrono::duration<double>{state.sine_valid ? 6.0 : 3.0};
  bool degraded = std::chrono::duration<double>(
    plan_time - state.start_timestamp) < convergence;
  for (int iteration = 0; iteration < 5; ++iteration) {
    if (degraded) break;
    auto future = state;
    const double prediction = fly_time + base_delay;
    future.transition(prediction);
    blade = -1;
    for (std::size_t i = 0; i < future.inactive.size(); ++i) {
      if (future.inactive[i]) {
        blade = static_cast<int>(i);
        break;
      }
    }
    if (blade < 0) {
      // 源 FireController 在收敛窗口或视野受限时仍然瞄准符中心，
      // 但把本帧标为 degraded 并禁止开火；不能把它误当作“无计划”。
      degraded = true;
      attack = result.center;
      attack_box_valid = false;
      break;
    }
    const AttackGeometry geometry = geometryAt(future, blade);
    attack = geometry.center;
    attack_box = geometry.box;
    attack_box_valid = true;
    const auto trajectory = trajectory_.solve(attack, bullet_speed.value);
    if (!trajectory) break;
    const double next_fly = trajectory->fly_time;
    if (std::abs(next_fly - fly_time) < 0.001) {
      fly_time = next_fly;
      break;
    }
    fly_time = next_fly;
  }

  const auto trajectory = trajectory_.solve(attack, bullet_speed.value);
  if (!trajectory) {
    result.degraded = degraded;
    result.pre_aim = !scheduler_.started();
    return result;
  }
  result.valid = true;
  result.degraded = degraded;
  result.attack = attack;
  result.attack_box = attack_box;
  result.attack_box_valid = attack_box_valid && !degraded && blade >= 0;
  result.blade_index = blade;
  result.yaw = trajectory->yaw + config_.offset_yaw_deg * std::numbers::pi / 180.0;
  result.pitch = trajectory->pitch + config_.offset_pitch_deg * std::numbers::pi / 180.0;
  result.target_direction = Eigen::Vector3d{
    std::cos(result.pitch) * std::cos(result.yaw),
    std::cos(result.pitch) * std::sin(result.yaw),
    -std::sin(result.pitch)};
  result.fly_time = trajectory->fly_time;
  result.prediction_time = plan_time + std::chrono::duration_cast<TimePoint::duration>(
    std::chrono::duration<double>{config_.shoot_delay + result.fly_time});

  if (!scheduler_.started()) scheduler_.start(plan_time);
  result.pre_aim = !scheduler_.tick(plan_time);
  // 回退弹速仍可生成跟随角，但除非配置明确授权，否则不能进入开火节拍。
  result.algorithm_shoot =
    !result.degraded && !result.pre_aim && bullet_speed.trusted;
  return result;
}

}  // namespace L4Planning::Buff
