#include "l4_planning/planner.hpp"

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/latency_compensator.hpp"
#include "l4_planning/predictor.hpp"
#include "l4_planning/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace L4Planning {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kOutpostRobotId = 6;
constexpr double kEnteringWindowLeadAngle = 10.0 * kPi / 180.0;

// 角度差统一归一化，保证跨越 ±pi 时仍取最短角距离。
[[nodiscard]] double normalizeAngle(double angle) noexcept
{
  angle = std::remainder(angle, 2.0 * kPi);
  return angle <= -kPi ? angle + 2.0 * kPi : angle;
}

// 用秒为单位向 steady_clock::time_point 增加预测时长。
[[nodiscard]] TimePoint addSeconds(TimePoint time, double seconds)
{
  return time + std::chrono::duration_cast<TimePoint::duration>(
                  std::chrono::duration<double>(seconds));
}

// 从每次整车预测产生的四块装甲板中保持同一 armor_id 继续迭代。
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

[[nodiscard]] bool validBulletSpeed(const L1Sensor::RobotState& robot_state) noexcept
{
  return std::isfinite(robot_state.bullet_speed)
         && robot_state.bullet_speed > 0.0;
}

[[nodiscard]] bool validGimbalOrientation(
  const L1Sensor::RobotState& robot_state) noexcept
{
  return std::isfinite(robot_state.rpy.yaw)
         && std::isfinite(robot_state.rpy.pitch);
}

// 配置校验在规划前集中完成，避免零容差进入后续计算。
[[nodiscard]] bool validPlannerConfig(const PlannerConfig& config) noexcept
{
  return config.max_iterations > 0
         && config.fly_time_tolerance.count() > 0
         && std::isfinite(config.position_tolerance)
         && config.position_tolerance > 0.0
         && std::isfinite(config.angle_tolerance)
         && config.angle_tolerance >= 0.0
         && std::isfinite(config.switch_dead_zone)
         && config.switch_dead_zone >= 0.0
         && std::isfinite(config.score_switch_threshold)
         && config.score_switch_threshold >= 0.0
         && config.score_switch_stable_frames > 0
         && std::isfinite(config.rotation_rate_dead_zone)
         && config.rotation_rate_dead_zone >= 0.0
         && config.lock_stable_frames > 0
         && std::isfinite(config.aim_cost_good_angle)
         && config.aim_cost_good_angle >= 0.0
         && std::isfinite(config.aim_cost_bad_angle)
         && config.aim_cost_bad_angle > config.aim_cost_good_angle
         && std::isfinite(config.normal_enter_angle)
         && config.normal_enter_angle > 0.0
         && std::isfinite(config.normal_leave_angle)
         && config.normal_leave_angle >= 0.0
         && config.normal_leave_angle <= config.normal_enter_angle
         && std::isfinite(config.outpost_enter_angle)
         && config.outpost_enter_angle > 0.0
         && std::isfinite(config.outpost_leave_angle)
         && config.outpost_leave_angle >= 0.0
         && config.outpost_leave_angle <= config.outpost_enter_angle
         && config.max_lost_frames >= 0;
}

// 重力和阻力系数由 PlannerConfig 统一管理。
[[nodiscard]] bool validBallisticConfig(
  const PlannerConfig& config) noexcept
{
  const bool drag_ready =
    !config.enable_air_resistance
    || (std::isfinite(config.linear_drag_coefficient)
        && config.linear_drag_coefficient > 0.0);
  return std::isfinite(config.gravity)
         && config.gravity > 0.0
         && drag_ready;
}

[[nodiscard]] double clampUnit(double value) noexcept
{
  return std::clamp(value, 0.0, 1.0);
}

[[nodiscard]] double smoothstep(double value) noexcept
{
  const double x = clampUnit(value);
  return x * x * (3.0 - 2.0 * x);
}

[[nodiscard]] bool validArmorScoreWeights(
  const ArmorScoreWeights& weights) noexcept
{
  if (!std::isfinite(weights.facing_weight)
      || !std::isfinite(weights.window_weight)
      || !std::isfinite(weights.aim_cost_weight)
      || weights.facing_weight < 0.0
      || weights.window_weight < 0.0
      || weights.aim_cost_weight < 0.0) {
    return false;
  }

  const double sum =
    weights.facing_weight + weights.window_weight + weights.aim_cost_weight;
  return std::abs(sum - 1.0) <= 1e-9;
}

[[nodiscard]] bool validFacingAngleThresholds(
  const PlannerContext& context) noexcept
{
  return std::isfinite(context.facing_angle_good)
         && context.facing_angle_good >= 0.0
         && std::isfinite(context.facing_angle_bad)
         && context.facing_angle_bad > context.facing_angle_good;
}

}  // namespace

Planner::Planner(PlannerConfig config)
  : config_(std::move(config))
{
}

void Planner::setConfig(PlannerConfig config)
{
  config_ = std::move(config);
  resetTracking();
}

const PlannerConfig& Planner::config() const noexcept
{
  return config_;
}

void Planner::resetTracking() noexcept
{
  tracking_state_ = {};
  last_target_.reset();
  last_observation_timestamp_.reset();
  target_lost_frames_ = 0;
}

const ArmorTrackingState& Planner::trackingState() const noexcept
{
  return tracking_state_;
}

AimPlan Planner::plan(
  const std::optional<L3Estimation::TargetState>& target,
  const L1Sensor::RobotState& robot_state)
{
  PlannerContext context;
  context.planning_time = robot_state.timestamp;
  context.config = config_;
  return plan(target, robot_state, context);
}

AimPlan Planner::plan(
  const std::optional<L3Estimation::TargetState>& target,
  const L1Sensor::RobotState& robot_state,
  const PlannerContext& context)
{
  const PlannerConfig& config = context.config;

  AimPlan plan;
  plan.generated_at = robot_state.timestamp;
  plan.using_MPC = false;

  if (!validBulletSpeed(robot_state)
      || !validGimbalOrientation(robot_state)) {
    return plan;
  }
  if (!validPlannerConfig(config)
      || !validBallisticConfig(config)
      || !validArmorScoreWeights(context.armor_score_weights)
      || !validFacingAngleThresholds(context)) {
    return plan;
  }
  if (!context.T_barrel_world.translation().allFinite()) {
    return plan;
  }

  const bool observation_fresh =
    target.has_value()
    && (!last_observation_timestamp_.has_value()
        || target->timestamp > *last_observation_timestamp_);
  if (observation_fresh) {
    if (tracking_state_.robot_id >= 0
        && tracking_state_.robot_id != target->robot_id) {
      tracking_state_ = {};
    }
    last_target_ = *target;
    last_observation_timestamp_ = target->timestamp;
    target_lost_frames_ = 0;
  } else {
    if (!last_target_.has_value()
        || target_lost_frames_ >= config.max_lost_frames) {
      resetTracking();
      return plan;
    }
    ++target_lost_frames_;
  }

  const L3Estimation::TargetState& target_state = *last_target_;
  tracking_state_.robot_id = target_state.robot_id;
  plan.target_id = target_state.robot_id;
  plan.tracking = true;
  plan.tracking_phase = tracking_state_.phase;

  // 图像/滤波时刻到规划时刻的耗时由时间戳计算，标定的出膛延迟由
  // context.latency 提供，两者均由 LatencyCompensator 统一校验。
  const LatencyCompensator latency_compensator{context.latency};
  const LatencyResult latency = latency_compensator.calculate(
    target_state.timestamp, robot_state.timestamp);
  if (!latency.valid) {
    return plan;
  }
  const TimePoint fire_time = addSeconds(
    latency.delay.camera_timestamp, latency.delay.total());

  const double fly_time_tolerance =
    std::chrono::duration<double>(config.fly_time_tolerance).count();
  const double angle_tolerance = config.angle_tolerance;
  const int max_iterations = std::max(1, config.max_iterations);
  // 世界系和枪口系轴向始终平行，世界系到枪口系只改变原点。
  // T_barrel_world 的平移用于将世界系位置转换到枪口系。
  const auto world_to_barrel =
    [&context](const Eigen::Vector3d& position_world) {
      return position_world + context.T_barrel_world.translation();
    };

  Predictor predictor;
  BallisticSolver ballistic_solver;
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
  candidates.reserve(4);

  // 与 calculateShootingParameters 相同，这里使用固定点迭代：
  //   旧飞行时间 -> 命中时刻的装甲板位置 -> 新飞行时间。
  // 区别是对四块装甲板分别迭代，并同时检查飞行时间和位置（或角度）
  // 的收敛性，未收敛的候选不会进入最终选择。
  for (int armor_id = 0; armor_id < 4; ++armor_id) {
    ArmorCandidate candidate;
    // current_fly_time 是固定点变量：本轮用它预测位置，再得到下一轮值。
    double current_fly_time = 0.0;
    // 以下三个量保存上一轮结果，用于计算位置和瞄准角收敛误差。
    Eigen::Vector3d previous_position =
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
    double previous_yaw = std::numeric_limits<double>::quiet_NaN();
    double previous_pitch = std::numeric_limits<double>::quiet_NaN();

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
      const TimePoint impact_time =
        addSeconds(fire_time, current_fly_time);
      const PredictionResult prediction =
        predictor.predict({target_state, impact_time});
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
      // 再按收敛后的飞行时间更新一次最终位置，避免原实现中“最终 pose
      // 已更新但 yaw/pitch 仍属于上一次 pose”的不一致。
      const TimePoint final_impact_time =
        addSeconds(fire_time, candidate.ballistic.fly_time);
      const PredictionResult final_prediction =
        predictor.predict({target_state, final_impact_time});
      const ArmorPose* final_armor = findArmor(final_prediction, armor_id);
      if (final_prediction.valid && final_armor != nullptr && final_armor->valid) {
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
      const double enter_angle = enter_angle_degree * kPi / 180.0;
      const double leave_angle = leave_angle_degree * kPi / 180.0;
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

      double window_quality = 0.0;
      if (std::abs(candidate.relative_yaw_rate)
          > config.rotation_rate_dead_zone) {
        const double rotation_direction =
          candidate.relative_yaw_rate > 0.0 ? 1.0 : -1.0;
        candidate.phase_angle = normalizeAngle(
          rotation_direction * candidate.delta_angle);
        candidate.within_firing_window =
          candidate.phase_angle >= -enter_angle
          && candidate.phase_angle <= leave_angle;
        // 预进入区位于正式射击窗口之前：从进入角外侧 10 degree
        // 到进入角边界。它不属于射击窗口，仅用于提前选择下一块板。
        candidate.entering_firing_window =
          candidate.phase_angle >= -(enter_angle + kEnteringWindowLeadAngle)
          && candidate.phase_angle < -enter_angle;
        candidate.remaining_window_time =
          candidate.within_firing_window
            ? std::max(
                0.0,
                (leave_angle - candidate.phase_angle)
                  / std::abs(candidate.relative_yaw_rate))
            : 0.0;
        const double normalized_window_progress =
          (candidate.phase_angle + enter_angle)
          / (enter_angle + leave_angle);
        window_quality =
          candidate.within_firing_window
            ? 1.0 - smoothstep(normalized_window_progress)
            : 0.0;
      } else {
        candidate.phase_angle = std::abs(candidate.delta_angle);
        candidate.within_firing_window =
          std::abs(candidate.delta_angle) <= enter_angle;
        candidate.entering_firing_window = false;
        candidate.remaining_window_time =
          candidate.within_firing_window
            ? std::numeric_limits<double>::infinity()
            : 0.0;
        window_quality = candidate.within_firing_window ? 1.0 : 0.0;
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
      const double aim_yaw_error = std::abs(normalizeAngle(
        candidate.ballistic.yaw - robot_state.rpy.yaw));
      const double aim_pitch_error =
        std::abs(candidate.ballistic.pitch - robot_state.rpy.pitch);
      candidate.aim_angle_error =
        std::hypot(aim_yaw_error, aim_pitch_error);
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

  SelectionRequest selection_request;
  selection_request.candidates = std::move(candidates);
  selection_request.preferred_armor_id =
    tracking_state_.current_armor_id;
  selection_request.observation_fresh = observation_fresh;
  const TimePoint selection_time =
    context.planning_time != TimePoint{}
      ? context.planning_time
      : robot_state.timestamp;
  const SelectionResult selection =
    selectArmor(selection_request, selection_time, config);
  plan.tracking_phase = selection.phase;
  if (selection.phase == ArmorTrackingPhase::Unlocked) {
    plan.tracking = false;
    plan.target_id = -1;
  }
  if (!selection.valid || !selection.selected.has_value()) {
    return plan;
  }
  const ArmorCandidate& selected = *selection.selected;

  const Eigen::Vector3d final_position_barrel =
    world_to_barrel(selected.armor.position_world);
  plan.armor_id = selected.armor.armor_id;
  plan.impact_time = selected.impact_time;
  plan.aim_point_barrel = final_position_barrel;
  plan.aim_point_world = selected.armor.position_world;
  plan.yaw = selected.ballistic.yaw;
  plan.pitch = selected.ballistic.pitch;
  plan.fly_time = selected.ballistic.fly_time;
  plan.fire_permitted =
    selection.tracking_ready
    && selected.within_firing_window;
  plan.valid = true;

  return plan;
}

}
