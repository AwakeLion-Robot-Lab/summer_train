#include "l4_planning/tiny_mpc.hpp"

#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/predictor.hpp"

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

[[nodiscard]] const ArmorPose* findArmor(
  const PredictionResult& prediction,
  int armor_id) noexcept
{
  const auto iterator = std::find_if(
    prediction.armor_candidates.begin(),
    prediction.armor_candidates.end(),
    [armor_id](const ArmorPose& armor) {
      return armor.armor_id == armor_id;
    });
  return iterator == prediction.armor_candidates.end() ? nullptr : &*iterator;
}

// Predictor 的绝对时刻入口只允许外推。居中窗口前部可能早于 L3 状态，
// 因此先用相同匀速模型回推到采样时刻，再在该时刻展开四块装甲板。
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
  BallisticSolver ballistic_solver;
  std::array<double, kTinyMpcHorizon> raw_yaw{};

  for (std::size_t index = 0; index < kTinyMpcHorizon; ++index) {
    const double offset =
      (static_cast<double>(index) - static_cast<double>(kTinyMpcControlIndex))
      * kTinyMpcStepSeconds;
    const TimePoint sample_time = addSeconds(direct_plan.impact_time, offset);
    const PredictionResult prediction = predictAt(predictor, target, sample_time);
    const ArmorPose* armor = findArmor(prediction, direct_plan.armor_id);
    if (!prediction.valid || armor == nullptr || !armor->valid) {
      return reference;
    }

    BallisticRequest request;
    request.target_position_barrel =
      armor->position_world + T_barrel_world.translation();
    request.bullet_speed = robot_state.bullet_speed;
    request.gravity = config.gravity;
    request.enable_air_resistance = config.enable_air_resistance;
    request.linear_drag_coefficient = config.linear_drag_coefficient;
    const BallisticSolution ballistic = ballistic_solver.solve(request);
    if (!ballistic.valid || !std::isfinite(ballistic.yaw)
        || !std::isfinite(ballistic.pitch)) {
      return reference;
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
