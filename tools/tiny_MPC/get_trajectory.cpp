#include "l3_planning/tiny_mpc.hpp"

#include "l3_planning/predictor.hpp"
#include "l3_planning/armor_candidate_generator.hpp"
#include "l3_planning/planner.hpp"
#include "l3_planning/in_plate_aim.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace L4Planning {

namespace {

constexpr double kPi = 3.14159265358979323846;

[[nodiscard]] double normalizeAngle(double angle) noexcept
{
  angle = std::remainder(angle, 2.0 * kPi);
  return angle <= -kPi ? angle + 2.0 * kPi : angle;
}

[[nodiscard]] TimePoint addSeconds(TimePoint time, double seconds)
{
  return time + std::chrono::duration_cast<TimePoint::duration>(
                  std::chrono::duration<double>(seconds));
}

// Predictor 的绝对时刻入口只允许外推。居中窗口前部可能早于 L3 状态，
// 因此先用相同匀速模型回推到采样时刻，再在该时刻展开三块或四块装甲板。
[[nodiscard]] PredictionResult predictAt(
  const Predictor& predictor,
  const L3Estimation::TargetState& target,
  TimePoint sample_time)
{
  if (sample_time >= target.timestamp) {
    return predictor.predict({target, sample_time});
  }

  const double dt = std::chrono::duration<double>(
    sample_time - target.timestamp).count();
  L3Estimation::TargetState rewound = predictor.predict(target, dt);
  rewound.timestamp = sample_time;
  return predictor.predict({rewound, sample_time});
}

}  // namespace

TinyMpcReference buildTinyMpcReference(
  const AimPlan& direct_plan,
  const L3Estimation::TargetState& target,
  const L1Sensor::RobotState& robot_state,
  const Eigen::Isometry3d& T_barrel_world,
  const PlannerConfig& config)
{
  TinyMpcReference reference;
  if (!direct_plan.valid || direct_plan.armor_id < 0
      || !std::isfinite(direct_plan.yaw)
      || !std::isfinite(robot_state.bullet_speed)
      || robot_state.bullet_speed <= 0.0
      || !T_barrel_world.translation().allFinite()) {
    return reference;
  }

  Predictor predictor;
  PlannerContext context;
  context.config = config;
  context.T_barrel_world = T_barrel_world;
  std::array<double, kTinyMpcHorizon> raw_yaw{};
  int reference_armor_id = direct_plan.armor_id;
  InPlateAimState rollout_state = direct_plan.aim_query_state;

  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    const double offset =
      (static_cast<double>(index) - static_cast<double>(kTinyMpcControlIndex))
      * kTinyMpcStepSeconds;
    // 每个采样点对应一个发射时刻，独立迭代飞行时间；中心点与直瞄完全同源。
    const TimePoint base_fire_time = direct_plan.armor_geometry_valid
      ? direct_plan.fire_time : addSeconds(direct_plan.impact_time, -direct_plan.fly_time);
    const TimePoint sample_fire_time = addSeconds(base_fire_time, offset);
    auto sample_target = target;
    if (sample_fire_time < target.timestamp) {
      const auto rewound = predictAt(predictor, target, sample_fire_time);
      if (!rewound.valid) return reference;
      sample_target = rewound.predicted_vehicle;
    }
    const InPlateAimState& query_state = index > kTinyMpcControlIndex
      ? rollout_state : direct_plan.aim_query_state;
    auto sample_robot = robot_state;
    if (index > kTinyMpcControlIndex) {
      sample_robot.rpy.yaw = normalizeAngle(robot_state.rpy.yaw
        + direct_plan.aim_query_state.gun_yaw_rate * offset);
    }
    auto candidates = generateArmorCandidates(
      sample_target, sample_robot, context, sample_fire_time,
      query_state, reference_armor_id);
    if (!candidates || candidates->empty()) return reference;
    ArmorCandidate selected = candidates->front();
    const bool rotating = selected.aim.direction != 0;
    const bool leaving_hold = !selected.within_selection_hold_window
      && (!rotating || selected.phase_angle > 0.0);
    if (index > kTinyMpcControlIndex && (!selected.valid || leaving_hold)) {
      // 未来采样按保持窗口预测接替板，仅改变本次参考的局部板号，绝不更新 Planner。
      candidates = generateArmorCandidates(sample_target, sample_robot, context,
        sample_fire_time, query_state);
      if (!candidates) return reference;
      const ArmorCandidate* next = nullptr;
      if (rotating) {
        const int next_id = nextArmorInRotation(reference_armor_id, target.armor_count, selected.aim.direction);
        const auto cyclic_next = std::find_if(candidates->begin(), candidates->end(),
          [next_id](const ArmorCandidate& candidate) {
            return candidate.armor.armor_id == next_id && candidate.valid;
          });
        // 窗口间可能存在空隙，参考可提前转向接替板；实际落点检查负责禁火。
        if (cyclic_next != candidates->end()) next = &*cyclic_next;
      }
      const bool cyclic_next_found = next != nullptr;
      for (const auto& candidate : *candidates) {
        if (!cyclic_next_found && candidate.valid && candidate.within_selection_window
            && (!next || candidate.score.quality > next->score.quality)) {
          next = &candidate;
        }
      }
      if (next) {
        selected = *next;
        reference_armor_id = selected.armor.armor_id;
      }
    }
    if (!selected.valid) return reference;
    const auto& ballistic = selected.ballistic;

    reference.armor_id[index] = selected.armor.armor_id;
    reference.in_plate_aim[index] = selected.aim;
    reference.impact_time[index] = selected.impact_time;
    if (index >= kTinyMpcControlIndex) {
      // 仅提交到本次参考的局部副本，每个未来采样推进一次，候选/弹道迭代只读。
      rollout_state.robot_id = target.robot_id;
      rollout_state.armor_id = selected.armor.armor_id;
      rollout_state.timestamp = sample_fire_time;
      rollout_state.offset = selected.aim.offset;
      rollout_state.velocity = selected.aim.velocity;
      rollout_state.position_world = selected.aim.position_world;
      rollout_state.direction = selected.aim.direction;
    }
    raw_yaw[index] = ballistic.yaw;
    reference.pitch.angle[index] = ballistic.pitch;
  }

  reference.yaw_origin = direct_plan.yaw;
  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    reference.yaw.angle[index] = normalizeAngle(
      raw_yaw[index] - reference.yaw_origin);
  }

  const auto differentiate = [](const auto& angle, auto& velocity, bool yaw) {
    const auto delta = [yaw](double right, double left) {
      return yaw ? normalizeAngle(right - left) : right - left;
    };
    velocity.front() =
      delta(angle[1], angle[0]) / kTinyMpcStepSeconds;
    for (std::size_t index = 1; index + 1 < kTinyMpcHorizon; ++index) {
      velocity[index] = delta(angle[index + 1], angle[index - 1])
                        / (2.0 * kTinyMpcStepSeconds);
    }
    velocity.back() =
      delta(angle.back(), angle[kTinyMpcHorizon - 2])
      / kTinyMpcStepSeconds;
  };
  differentiate(reference.yaw.angle, reference.yaw.angular_velocity, true);
  differentiate(reference.pitch.angle, reference.pitch.angular_velocity, false);

  reference.valid = true;
  return reference;
}

}  // namespace L4Planning
