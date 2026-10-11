#include "l4_planning/armor_candidate_generator.hpp"

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/target_state.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/planner.hpp"
#include "l4_planning/predictor.hpp"
#include "l4_planning/window_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace L4Planning {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kOutpostRobotId = 6;
constexpr double kEnteringSelectionLeadAngle = 10.0 * kPi / 180.0;

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

[[nodiscard]] double smoothstep(double value) noexcept
{
  const double x = std::clamp(value, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

}  // namespace

std::optional<std::vector<ArmorCandidate>> generateArmorCandidates(
  const L3Estimation::TargetState& target_state,
  const L1Sensor::RobotState& robot_state,
  const PlannerContext& context,
  TimePoint fire_time)
{
  const PlannerConfig& config = context.config;
  const double fly_time_tolerance =
    std::chrono::duration<double>(config.fly_time_tolerance).count();
  const double angle_tolerance = config.angle_tolerance;
  const int max_iterations = std::max(1, config.max_iterations);

  const auto world_to_barrel =
    [&context](const Eigen::Vector3d& position_world) {
      return position_world + context.T_barrel_world.translation();
    };

  Predictor predictor;
  BallisticSolver ballistic_solver;
  // 先把目标状态推进到预计出膛时刻，再从同一发射状态分别预测各装甲板。
  const PredictionResult launch_prediction =
    predictor.predict({target_state, fire_time});
  if (!launch_prediction.valid) {
    return std::nullopt;
  }
  const L3Estimation::TargetState& launch_state =
    launch_prediction.predicted_vehicle;
  const auto solve_ballistic =
    [&ballistic_solver, &robot_state, &config](
      const Eigen::Vector3d& position_barrel) {
      BallisticRequest request;
      request.target_position_barrel = position_barrel;
      request.bullet_speed = robot_state.bullet_speed;
      request.gravity = config.gravity;
      request.enable_air_resistance = config.enable_air_resistance;
      request.linear_drag_coefficient = config.linear_drag_coefficient;
      return ballistic_solver.solve(request);
    };

  std::vector<ArmorCandidate> candidates;
  candidates.reserve(static_cast<std::size_t>(target_state.armor_count));

  for (int armor_id = 0; armor_id < target_state.armor_count; ++armor_id) {
    ArmorCandidate candidate;
    double current_fly_time = 0.0;
    Eigen::Vector3d previous_position =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    double previous_yaw = std::numeric_limits<double>::quiet_NaN();
    double previous_pitch = std::numeric_limits<double>::quiet_NaN();

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
      const TimePoint impact_time =
        addSeconds(fire_time, current_fly_time);
      const PredictionResult prediction =
        predictor.predict({launch_state, impact_time});
      if (!prediction.valid) {
        break;
      }

      const ArmorPose* armor = findArmor(prediction, armor_id);
      if (armor == nullptr || !armor->valid) {
        break;
      }

      const Eigen::Vector3d position_barrel =
        world_to_barrel(armor->position_world);
      const BallisticSolution ballistic = solve_ballistic(position_barrel);

      candidate.armor = *armor;
      candidate.ballistic = ballistic;
      candidate.impact_time = impact_time;
      candidate.iteration_count = iteration;
      if (!ballistic.valid) {
        break;
      }
      candidate.fly_time_error =
        std::abs(ballistic.fly_time - current_fly_time);
      const bool has_previous = previous_position.allFinite();
      candidate.position_error =
        has_previous
          ? (armor->position_world - previous_position).norm()
          : std::numeric_limits<double>::infinity();
      const double yaw_error =
        has_previous
          ? normalizeAngle(ballistic.yaw - previous_yaw)
          : std::numeric_limits<double>::infinity();
      const double pitch_error =
        has_previous
          ? ballistic.pitch - previous_pitch
          : std::numeric_limits<double>::infinity();
      candidate.angle_error =
        has_previous
          ? std::hypot(yaw_error, pitch_error)
          : std::numeric_limits<double>::infinity();

      const bool time_converged =
        candidate.fly_time_error <= fly_time_tolerance;
      const bool position_converged =
        has_previous && candidate.position_error <= config.position_tolerance;
      const bool angle_converged =
        angle_tolerance > 0.0
        && has_previous
        && candidate.angle_error <= angle_tolerance;
      if (time_converged && (position_converged || angle_converged)) {
        candidate.converged = true;
        break;
      }

      previous_position = armor->position_world;
      previous_yaw = ballistic.yaw;
      previous_pitch = ballistic.pitch;
      current_fly_time = ballistic.fly_time;
    }

    if (candidate.converged) {
      // 用收敛后的飞行时间再更新一次最终位姿和弹道，保持二者一致。
      const TimePoint final_impact_time =
        addSeconds(fire_time, candidate.ballistic.fly_time);
      const PredictionResult final_prediction =
        predictor.predict({launch_state, final_impact_time});
      const ArmorPose* final_armor = findArmor(final_prediction, armor_id);
      if (final_prediction.valid
          && final_armor != nullptr
          && final_armor->valid) {
        const Eigen::Vector3d final_position_barrel =
          world_to_barrel(final_armor->position_world);
        const BallisticSolution final_ballistic =
          solve_ballistic(final_position_barrel);
        if (final_ballistic.valid) {
          const double final_yaw_error = normalizeAngle(
            final_ballistic.yaw - candidate.ballistic.yaw);
          const double final_pitch_error =
            final_ballistic.pitch - candidate.ballistic.pitch;
          candidate.position_error =
            (final_armor->position_world - candidate.armor.position_world).norm();
          candidate.fly_time_error =
            std::abs(final_ballistic.fly_time - candidate.ballistic.fly_time);
          candidate.angle_error =
            std::hypot(final_yaw_error, final_pitch_error);
          candidate.armor = *final_armor;
          candidate.ballistic = final_ballistic;
          candidate.impact_time = final_impact_time;

          const bool final_time_converged =
            candidate.fly_time_error <= fly_time_tolerance;
          const bool final_position_converged =
            candidate.position_error <= config.position_tolerance;
          const bool final_angle_converged =
            angle_tolerance > 0.0
            && candidate.angle_error <= angle_tolerance;
          candidate.converged =
            final_time_converged
            && (final_position_converged || final_angle_converged);
        } else {
          candidate.converged = false;
        }
      } else {
        candidate.converged = false;
      }
    }

    if (candidate.armor.valid) {
      const double prediction_dt = std::chrono::duration<double>(
        candidate.impact_time - target_state.timestamp).count();
      const Eigen::Vector3d predicted_center =
        target_state.center + target_state.velocity * prediction_dt;
      const double center_yaw = std::atan2(
        predicted_center.y(), predicted_center.x());
      candidate.delta_angle =
        normalizeAngle(candidate.armor.yaw_world - center_yaw);

      const double enter_angle_degree =
        target_state.robot_id == kOutpostRobotId
          ? config.outpost_enter_angle
          : config.normal_enter_angle;
      const double leave_angle_degree =
        target_state.robot_id == kOutpostRobotId
          ? config.outpost_leave_angle
          : config.normal_leave_angle;
      const double horizontal_distance_squared =
        predicted_center.x() * predicted_center.x()
        + predicted_center.y() * predicted_center.y();
      const double center_bearing_rate =
        horizontal_distance_squared > 1e-12
          ? (predicted_center.x() * target_state.velocity.y()
             - predicted_center.y() * target_state.velocity.x())
              / horizontal_distance_squared
          : 0.0;
      candidate.relative_yaw_rate =
        target_state.yaw_rate - center_bearing_rate;
      const DynamicWindows windows = computeDynamicWindows(
        enter_angle_degree,
        leave_angle_degree,
        candidate.relative_yaw_rate,
        config);

      double window_quality = 0.0;
      if (std::abs(candidate.relative_yaw_rate)
          > config.rotation_rate_dead_zone) {
        const double rotation_direction =
          candidate.relative_yaw_rate > 0.0 ? 1.0 : -1.0;
        candidate.phase_angle = normalizeAngle(
          rotation_direction * candidate.delta_angle);
        candidate.within_selection_window =
          candidate.phase_angle >= -windows.selection.enter
          && candidate.phase_angle <= windows.selection.leave;
        candidate.within_selection_hold_window =
          candidate.phase_angle >= -windows.hold.enter
          && candidate.phase_angle <= windows.hold.leave;
        candidate.within_firing_window =
          candidate.phase_angle >= -windows.firing.enter
          && candidate.phase_angle <= windows.firing.leave;
        candidate.entering_selection_window =
          candidate.phase_angle
            >= -(windows.selection.enter + kEnteringSelectionLeadAngle)
          && candidate.phase_angle < -windows.selection.enter;
        candidate.remaining_window_time =
          candidate.within_selection_window
            ? std::max(
                0.0,
                (windows.selection.leave - candidate.phase_angle)
                  / std::abs(candidate.relative_yaw_rate))
            : 0.0;
        const double normalized_window_progress =
          (candidate.phase_angle + windows.selection.enter)
          / (windows.selection.enter + windows.selection.leave);
        window_quality =
          candidate.within_selection_window
            ? 1.0 - smoothstep(normalized_window_progress)
            : 0.0;
      } else {
        candidate.phase_angle = std::abs(candidate.delta_angle);
        candidate.within_selection_window =
          candidate.phase_angle <= windows.selection.enter;
        candidate.within_selection_hold_window =
          candidate.phase_angle <= windows.hold.enter;
        candidate.within_firing_window =
          candidate.phase_angle <= windows.firing.enter;
        candidate.entering_selection_window = false;
        candidate.remaining_window_time =
          candidate.within_selection_window
            ? std::numeric_limits<double>::infinity()
            : 0.0;
        window_quality = candidate.within_selection_window ? 1.0 : 0.0;
      }

      const double facing_angle_good =
        context.facing_angle_good * kPi / 180.0;
      const double facing_angle_bad =
        context.facing_angle_bad * kPi / 180.0;
      const double normalized_facing_angle =
        (std::abs(candidate.delta_angle) - facing_angle_good)
        / (facing_angle_bad - facing_angle_good);
      const double facing_quality =
        1.0 - smoothstep(normalized_facing_angle);
      candidate.aim_yaw_error = std::abs(normalizeAngle(
        candidate.ballistic.yaw - robot_state.rpy.yaw));
      // 弹道 pitch 向上为正，下位机右手系 Ry pitch 向下为正。
      candidate.aim_pitch_error =
        std::abs(candidate.ballistic.pitch + robot_state.rpy.pitch);
      candidate.aim_angle_error =
        std::hypot(candidate.aim_yaw_error, candidate.aim_pitch_error);
      const double aim_cost_good_angle =
        config.aim_cost_good_angle * kPi / 180.0;
      const double aim_cost_bad_angle =
        config.aim_cost_bad_angle * kPi / 180.0;
      const double normalized_aim_cost =
        (candidate.aim_angle_error - aim_cost_good_angle)
        / (aim_cost_bad_angle - aim_cost_good_angle);
      const double aim_cost_quality =
        1.0 - smoothstep(normalized_aim_cost);

      candidate.score.components.Q_facing = facing_quality;
      candidate.score.components.Q_window = window_quality;
      candidate.score.components.Q_aim_cost = aim_cost_quality;
      candidate.score.hard_conditions.identity_consistent =
        candidate.armor.robot_id == target_state.robot_id;
      candidate.score.hard_conditions.stable_tracking =
        target_state.covariance.allFinite();
      candidate.score.hard_conditions.prediction_valid =
        candidate.armor.valid;
      candidate.score.hard_conditions.within_selection_window =
        candidate.within_selection_window;
      candidate.score.hard_conditions.within_firing_window =
        candidate.within_firing_window;
      candidate.score.hard_conditions.ballistic_valid =
        candidate.ballistic.valid;
      candidate.score.hard_conditions.iteration_converged =
        candidate.converged;
      const auto& hard = candidate.score.hard_conditions;
      candidate.valid =
        hard.identity_consistent
        && hard.stable_tracking
        && hard.prediction_valid
        && hard.ballistic_valid
        && hard.iteration_converged;
      const ArmorScoreWeights& weights = context.armor_score_weights;
      candidate.score.quality =
        weights.facing_weight * candidate.score.components.Q_facing
        + weights.window_weight * candidate.score.components.Q_window
        + weights.aim_cost_weight * candidate.score.components.Q_aim_cost;
    }

    candidates.push_back(candidate);
  }

  return candidates;
}

}  // namespace L4Planning
