#include "runtime/rune_runtime.hpp"

#include <cmath>
#include <utility>

namespace runtime {
namespace {
Rune::PlanningConfig planningConfig(const RuneRuntimeConfig& config,
                                    const L4Planning::PlannerTuning& tuning) {
  auto result = config.planning;
  result.gravity_m_s2 = tuning.planner.gravity;
  result.enable_air_resistance = tuning.planner.enable_air_resistance;
  result.linear_drag_coefficient = tuning.planner.linear_drag_coefficient;
  result.max_iterations = tuning.planner.max_iterations;
  result.flight_time_tolerance_s =
      std::chrono::duration<double>(tuning.planner.fly_time_tolerance).count();
  result.mpc = tuning.planner;
  if (config.mpc_enable) result.mpc.enable_mpc = *config.mpc_enable;
  return result;
}
}  // namespace

RuneRuntime::RuneRuntime(const RuneRuntimeConfig& config,
                         const L4Planning::PlannerTuning& tuning,
                         std::unique_ptr<Rune::IKeypointDetector> detector)
    : detector_(detector ? std::move(detector)
                         : std::make_unique<Rune::OpenVinoKeypointDetector>(
                               Rune::KeypointDetectorConfig::load(config.detector_config))),
      processor_(config.geometry, {}, planningConfig(config, tuning), {}, config.fire),
      tuning_(tuning), max_feedback_age_s_(config.fire.max_plan_age_s) {}

void RuneRuntime::detect(Rune::Frame& frame) {
  frame.detections = detector_->detect(frame.bgr, frame.mode);
}

std::optional<L1Sensor::SerialCommand> RuneRuntime::process(
    Rune::Frame frame, const L1Sensor::RobotState& state,
    const std::optional<Eigen::Quaterniond>& actual_pose,
    Rune::TimePoint command_at) {
  Rune::ControlInput control;
  control.command_at = command_at;
  control.feedback_at = state.timestamp;
  control.bullet_speed_m_s = std::isfinite(state.bullet_speed) && state.bullet_speed > 0
      ? state.bullet_speed : tuning_.default_bullet_speed;
  control.fire_delay_s = tuning_.latency.fireDelay(command_.yaw_rate_rad_s);
  const bool actual_pose_valid = actual_pose && actual_pose->coeffs().allFinite() &&
      actual_pose->norm() > 1e-6;
  if (actual_pose_valid) {
    const Eigen::Vector3d forward = actual_pose->normalized() * Eigen::Vector3d::UnitX();
    control.actual_yaw_rad = std::atan2(forward.y(), forward.x());
    control.actual_pitch_rad = std::atan2(forward.z(), std::hypot(forward.x(), forward.y()));
  }
  frame.color = state.enemy_color == L1Sensor::EnemyColor::Blue
      ? Rune::Color::Blue : Rune::Color::Red;
  // 没有曝光姿态、标定或目标颜色时，不能把相机相对角度当作绝对指令。
  const double feedback_age_s = std::chrono::duration<double>(command_at - state.timestamp).count();
  if (!frame.q_world_barrel || !frame.q_world_barrel->coeffs().allFinite() ||
      frame.q_world_barrel->norm() < 1e-6 || !actual_pose_valid ||
      !frame.T_barrel_camera || frame.camera_matrix.empty() ||
      state.timestamp == Rune::TimePoint{} || feedback_age_s < 0 ||
      feedback_age_s > max_feedback_age_s_ ||
      state.enemy_color == L1Sensor::EnemyColor::Unknown) {
    frame.detections.clear();
  }
  command_ = processor_.process(frame, control);
  if (auto command = Rune::ControlStage::makeSerialCommand(command_)) return command;
  return processor_.safeHold();
}

void RuneRuntime::suspend() {
  Rune::Frame frame;
  frame.mode = command_.mode;
  command_ = processor_.process(frame, {});
}

}  // namespace runtime
