#pragma once

#include "l4_planning/ballistic_solver.hpp"
#include "l4_planning/predictor.hpp"
#include "l4_planning/types.hpp"

#include <optional>
#include <vector>

namespace L4Planning {

// 装甲板质量评分的权重。三项权重之和为 1。
struct ArmorScoreWeights {
  double facing_weight{0.40};
  double window_weight{0.40};
  double aim_cost_weight{0.20};
};

// 三项归一化质量分量，取值范围均为 [0, 1]。
// Q_aim_cost 越大表示转向代价越小。
struct ArmorScoreComponents {
  double Q_facing{0.0};
  double Q_window{0.0};
  double Q_aim_cost{0.0};
};

struct ArmorScoreHardConditions {
  bool identity_consistent{false};
  bool stable_tracking{false};
  bool prediction_valid{false};
  bool within_firing_window{false};
  bool ballistic_valid{false};
  bool iteration_converged{false};
};

// 一块候选装甲板的完整评分结果。score 只表示质量，不承担开火门控。
struct ArmorScore {
  ArmorScoreComponents components;
  ArmorScoreHardConditions hard_conditions;
  double quality{0.0};
};

struct ArmorCandidate {
  ArmorPose armor;                 // 迭代收敛后的装甲板世界系位姿
  BallisticSolution ballistic;    // 对最终装甲板位置的弹道解
  TimePoint impact_time{};        // 预测弹丸命中装甲板的绝对时刻
  double delta_angle{0.0};        // 装甲板法向与目标方位角之差，rad
  int iteration_count{0};         // 固定点迭代实际执行次数
  double fly_time_error{0.0};     // 相邻两次飞行时间之差，s
  double position_error{0.0};     // 相邻两次预测位置之差，m
  double angle_error{0.0};        // 相邻两次瞄准角的二维误差，rad
  double aim_angle_error{0.0};    // 当前云台到候选弹道角的合成角差，rad
  double relative_yaw_rate{0.0};  // 装甲板法线相对目标方位的角速度，rad/s
  double phase_angle{0.0};        // 沿旋转方向递增的窗口相位，rad
  double remaining_window_time{0.0}; // 到离开射击窗口的预计时间，s
  bool entering_firing_window{false}; // 是否位于车辆中心前的转入区间
  bool converged{false};          // 时间误差和位置/角度误差是否收敛
  bool within_firing_window{false}; // 命中时刻是否仍在可射击窗口
  bool valid{false};              // 除射击窗口外的跟踪、预测和弹道条件有效
  ArmorScore score;               // 用于多装甲板选择的最终评分
};

enum class SelectionReason {
  NoCandidate,
  InitialLock,
  KeepCurrent,
  SwitchToCandidate,
  Stabilizing,
  LockConfirmed,
  PreferredArmor,
  EnteringArmor,
  MostFacing,
  BallisticFeasible
};

struct SelectionRequest {
  std::vector<ArmorCandidate> candidates;
  std::optional<int> preferred_armor_id;
  bool observation_fresh{true};
};

struct SelectionResult {
  std::optional<ArmorCandidate> selected;
  ArmorTrackingPhase phase{ArmorTrackingPhase::Unlocked};
  // 已完成稳定锁定，且本周期观测足以支持跟踪；不包含射击窗口判断。
  bool tracking_ready{false};
  bool valid{false};
  SelectionReason reason{SelectionReason::NoCandidate};
};

struct ArmorTrackingState {
  ArmorTrackingPhase phase{ArmorTrackingPhase::Unlocked};
  int robot_id{-1};
  std::optional<int> current_armor_id;
  std::optional<int> next_armor_id;
  TimePoint phase_started_at{};
  int current_lost_frames{0};
  int next_stable_frames{0};
  std::optional<int> score_candidate_id;
  int score_stable_frames{0};
};

}  // namespace L4Planning
