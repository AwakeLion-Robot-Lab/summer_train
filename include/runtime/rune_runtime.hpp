#pragma once

#include "armor/l3_planning/planner_config.hpp"
#include "rune/f1-observation/keypoint_detector.hpp"
#include "rune/processor.hpp"
#include "sensor/serial/robot_state.hpp"

namespace runtime {

struct RuneRuntimeConfig {
  std::filesystem::path detector_config{"config/rune_detector.yaml"};
  Rune::Geometry geometry;
  Rune::PlanningConfig planning;
  Rune::FireControlConfig fire;
  // 未配置时沿用 planner_config.yaml 的 MPC 开关及参数。
  std::optional<bool> mpc_enable;
};

// 只负责传感器数据适配；大小能量机关运动模型仍由 Processor 按 Frame::mode 分流。
class RuneRuntime {
public:
  RuneRuntime(const RuneRuntimeConfig& config,
              const L4Planning::PlannerTuning& tuning,
              std::unique_ptr<Rune::IKeypointDetector> detector = {});
  void detect(Rune::Frame& frame);
  [[nodiscard]] std::optional<L1Sensor::SerialCommand> process(
      Rune::Frame frame, const L1Sensor::RobotState& state,
      const std::optional<Eigen::Quaterniond>& actual_pose,
      Rune::TimePoint command_at);
  // 丢失传感器输入时中断跟踪，保留开火冷却。
  void suspend();
  [[nodiscard]] const Rune::Command& lastCommand() const { return command_; }

private:
  std::unique_ptr<Rune::IKeypointDetector> detector_;
  Rune::Processor processor_;
  L4Planning::PlannerTuning tuning_;
  double max_feedback_age_s_;
  Rune::Command command_;
};

}  // namespace runtime
