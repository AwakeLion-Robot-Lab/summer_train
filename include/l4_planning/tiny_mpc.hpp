#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"

#include <Eigen/Geometry>

#include <array>
#include <cstddef>
#include <vector>

namespace L4Planning {

inline constexpr std::size_t kTinyMpcHorizon = 100;
inline constexpr std::size_t kTinyMpcControlIndex = 50;
inline constexpr double kTinyMpcStepSeconds = 0.01;

struct TinyMpcAxisReference {
  std::array<double, kTinyMpcHorizon> angle{};
  std::array<double, kTinyMpcHorizon> angular_velocity{};
};

struct TinyMpcReference {
  TinyMpcAxisReference yaw;
  TinyMpcAxisReference pitch;
  // yaw 以中心瞄准角为原点求解，输出时再加回并归一化。
  double yaw_origin{0.0};
  bool valid{false};
};

struct TinyMpcAxisConfig {
  double angle_weight{0.0};
  double velocity_weight{0.0};
  double acceleration_weight{0.0};
  double min_acceleration{0.0};
  double max_acceleration{0.0};
  double rho{0.0};
  int max_iterations{0};
  double primal_tolerance{0.0};
  double dual_tolerance{0.0};
};

struct TinyMpcSolveInfo {
  int iteration_count{0};
  double primal_residual{0.0};
  double dual_residual{0.0};
  bool converged{false};
};

struct TinyMpcAxisSolution {
  std::array<double, kTinyMpcHorizon> angle{};
  std::array<double, kTinyMpcHorizon> angular_velocity{};
  std::array<double, kTinyMpcHorizon> acceleration{};
  TinyMpcSolveInfo info;
  bool valid{false};
};

struct TinyMpcSolution {
  TinyMpcAxisSolution yaw;
  TinyMpcAxisSolution pitch;
  bool valid{false};
};

// 生成以 direct_plan.impact_time 为中心的 100 点理想瞄准轨迹。
[[nodiscard]] TinyMpcReference buildTinyMpcReference(
  const AimPlan& direct_plan,
  const L3Estimation::TargetState& target,
  const L1Sensor::RobotState& robot_state,
  const Eigen::Isometry3d& T_barrel_world,
  const PlannerConfig& config);

// 求解单轴有限时域二次规划。返回轨迹始终同时满足离散动力学和
// 加速度边界；valid 还要求 ADMM 达到配置残差阈值。
[[nodiscard]] TinyMpcAxisSolution solveTinyMpcAxis(
  const TinyMpcAxisReference& reference,
  const TinyMpcAxisConfig& config);

[[nodiscard]] TinyMpcSolution solveTinyMpc(
  const TinyMpcReference& reference,
  const PlannerConfig& config);

// 将第 50 点及其后的求解结果转换为按执行时间排序的 AimSample。
[[nodiscard]] std::vector<AimSample> makeTinyMpcSamples(
  const TinyMpcReference& reference,
  const TinyMpcSolution& solution,
  TimePoint first_execute_time);

// 在直瞄结果之上运行完整的参考生成、优化和协议输出。任一步失败时
// 原样返回 direct_plan（using_MPC=false、samples 为空）。
[[nodiscard]] AimPlan applyTinyMpc(
  const AimPlan& direct_plan,
  const L3Estimation::TargetState& target,
  const L1Sensor::RobotState& robot_state,
  const Eigen::Isometry3d& T_barrel_world,
  const PlannerConfig& config);

}  // namespace L4Planning
