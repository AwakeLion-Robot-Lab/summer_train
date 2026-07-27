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
  return std::isfinite(robot_state.rpy.roll)
         && std::isfinite(robot_state.rpy.pitch)
         && std::isfinite(robot_state.rpy.yaw);
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
         && std::isfinite(config.normal_enter_angle)
         && config.normal_enter_angle > 0.0
         && std::isfinite(config.outpost_enter_angle)
         && config.outpost_enter_angle > 0.0;
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

[[nodiscard]] Eigen::Matrix3d rotationGimbalToWorld(
  const L1Sensor::Orientation& orientation)
{
  return (
    Eigen::AngleAxisd(orientation.yaw, Eigen::Vector3d::UnitZ())
    * Eigen::AngleAxisd(orientation.pitch, Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(orientation.roll, Eigen::Vector3d::UnitX()))
    .toRotationMatrix();
}

[[nodiscard]] double clampUnit(double value) noexcept
{
  return std::clamp(value, 0.0, 1.0);
}

}  // namespace

Planner::Planner(PlannerConfig config)
  : config_(std::move(config))
{
}

void Planner::setConfig(PlannerConfig config)
{
  config_ = std::move(config);
}

const PlannerConfig& Planner::config() const noexcept
{
  return config_;
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

  if (!target.has_value()) {
    return plan;
  }

  plan.target_id = target->robot_id;
  plan.tracking = true;
  if (!validBulletSpeed(robot_state)) {
    return plan;
  }
  if (!validGimbalOrientation(robot_state)) {
    return plan;
  }
  if (!validPlannerConfig(config) || !validBallisticConfig(config)) {
    return plan;
  }

  // 当前接口尚未暴露独立的标定出膛延迟，因此这里先将 fire_delay 置零。
  // 图像/滤波时刻到本次规划时刻的延迟仍由 LatencyCompensator 统一校验。
  const Delay requested_delay{
    target->timestamp,
    robot_state.timestamp,
    0.0};
  const LatencyCompensator latency_compensator;
  const LatencyResult latency = latency_compensator.calculate(requested_delay);
  if (!latency.valid) {
    return plan;
  }
  const TimePoint fire_time = addSeconds(
    latency.delay.camera_timestamp, latency.delay.total());

  const double fly_time_tolerance =
    std::chrono::duration<double>(config.fly_time_tolerance).count();
  const double angle_tolerance = config.angle_tolerance;
  const int max_iterations = std::max(1, config.max_iterations);
  // 将世界系预测点转换到当前枪管坐标系后再进行弹道求解。
  const Eigen::Matrix3d world_to_barrel =
    rotationGimbalToWorld(robot_state.rpy).transpose();

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
        predictor.predict({*target, impact_time});
      if (!prediction.valid) {
        break;
      }

      const ArmorPose* armor = findArmor(prediction, armor_id);
      if (armor == nullptr || !armor->valid) {
        break;
      }

      // TargetState 的世界系原点位于云台旋转中心；当前 hpp 未提供额外
      // 平移外参，因此只需用实时云台姿态转到枪管坐标系。
      const Eigen::Vector3d position_barrel =
        world_to_barrel * armor->position_world;
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
        predictor.predict({*target, final_impact_time});
      const ArmorPose* final_armor = findArmor(final_prediction, armor_id);
      if (final_prediction.valid && final_armor != nullptr && final_armor->valid) {
        const Eigen::Vector3d final_position_barrel =
          world_to_barrel * final_armor->position_world;
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
      const double center_yaw = std::atan2(
        candidate.armor.position_world.y(),
        candidate.armor.position_world.x());
      candidate.delta_angle =
        normalizeAngle(candidate.armor.yaw_world - center_yaw);

      const double firing_angle_degree =
        target->robot_id == kOutpostRobotId
          ? config.outpost_enter_angle
          : config.normal_enter_angle;
      const double firing_angle = firing_angle_degree * kPi / 180.0;
      candidate.within_firing_window =
        std::abs(candidate.delta_angle) <= firing_angle;

      const double facing_quality = firing_angle > 0.0
        ? clampUnit(1.0 - std::abs(candidate.delta_angle) / firing_angle)
        : 0.0;
      const double covariance_quality = clampUnit(
        1.0 / (1.0
          + std::max(0.0, target->covariance(L3Estimation::XC, L3Estimation::XC))
          + std::max(0.0, target->covariance(L3Estimation::YC, L3Estimation::YC))
          + std::max(0.0, target->covariance(L3Estimation::ZC, L3Estimation::ZC))
          + std::max(0.0, target->covariance(L3Estimation::YAW, L3Estimation::YAW))));
      const double ballistic_quality =
        candidate.converged
          ? 0.5 * clampUnit(
              1.0 - candidate.fly_time_error / fly_time_tolerance)
            + 0.5 * clampUnit(
              1.0 - candidate.position_error / config.position_tolerance)
          : 0.0;

      candidate.score.components.Q_facing = facing_quality;
      // 第一版用窗口内的角度余量近似剩余射击窗口；后续可再结合
      // yaw_rate 和窗口穿越时间替换为仿真标定模型。
      candidate.score.components.Q_window = facing_quality;
      candidate.score.components.Q_prediction_confidence = covariance_quality;
      candidate.score.components.Q_ballistic = ballistic_quality;
      candidate.score.hard_conditions.identity_consistent =
        candidate.armor.robot_id == target->robot_id;
      candidate.score.hard_conditions.stable_tracking =
        target->covariance.allFinite();
      candidate.score.hard_conditions.prediction_valid =
        candidate.armor.valid;
      candidate.score.hard_conditions.within_firing_window =
        candidate.within_firing_window;
      candidate.score.hard_conditions.ballistic_valid =
        candidate.ballistic.valid;
      candidate.score.hard_conditions.iteration_converged =
        candidate.converged;
      const auto& hard = candidate.score.hard_conditions;
      candidate.score.flag =
        hard.identity_consistent
        && hard.stable_tracking
        && hard.prediction_valid
        && hard.within_firing_window
        && hard.ballistic_valid
        && hard.iteration_converged;

      const ArmorScoreWeights weights;
      candidate.score.quality =
        weights.facing_weight * candidate.score.components.Q_facing
        + weights.window_weight * candidate.score.components.Q_window
        + weights.prediction_confidence_weight
          * candidate.score.components.Q_prediction_confidence
        + weights.ballistic_weight * candidate.score.components.Q_ballistic;
      candidate.score.score =
        candidate.score.flag ? candidate.score.quality : 0.0;
      candidate.valid = candidate.score.flag;
    }

    candidates.push_back(candidate);
  }

  const auto selected = std::max_element(
    candidates.begin(),
    candidates.end(),
    [](const ArmorCandidate& lhs, const ArmorCandidate& rhs) {
      return lhs.score.score < rhs.score.score;
    });
  if (selected == candidates.end() || !selected->valid) {
    return plan;
  }

  const Eigen::Vector3d final_position_barrel =
    world_to_barrel * selected->armor.position_world;
  plan.impact_time = selected->impact_time;
  plan.aim_point_barrel = final_position_barrel;
  plan.aim_point_world = selected->armor.position_world;
  plan.yaw = selected->ballistic.yaw;
  plan.pitch = selected->ballistic.pitch;
  plan.fly_time = selected->ballistic.fly_time;
  plan.ballistic_valid = selected->ballistic.valid;
  plan.fire_permitted =
    selected->converged && selected->within_firing_window;
  plan.valid = true;

  return plan;
}

}  // namespace L4Planning
