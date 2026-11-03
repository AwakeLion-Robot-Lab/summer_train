#pragma once

#include <Eigen/Core>

#include <chrono>
#include <cstdint>
#include <vector>

namespace L4Planning {

using TimePoint = std::chrono::steady_clock::time_point;

enum class PlanType : std::uint8_t {
  Direct,
  Setpoint,
  QuinticSwitch,
  TinyMpc
};

enum class AimPlanStatus : std::uint8_t {
  NoTarget,
  Tracking,
  Switching,
  Ready,
  Failed
};

// 装甲板选择器的跨帧状态。Unlocked 不属于切换过程，其余三项对应
// 当前锁定、切换中和新装甲板稳定确认。
enum class ArmorTrackingPhase : std::uint8_t {
  Unlocked,
  Tracking,
  Switching,
  Stabilizing
};

enum class PlanError : std::uint8_t {
  None,
  NoTarget,
  NotTracking,
  NoArmor,
  NoCalibration,
  BadBulletSpeed,
  BallisticFailed,
  NotConverged,
  OutOfRange,
  Switching
};

// 一次规划实际使用的延迟结果，单位统一为秒。
  struct Delay {
    TimePoint camera_timestamp{};
    TimePoint command_timestamp{};
    double fire_delay{0.0};

    [[nodiscard]] double beforeFire() const noexcept
    {
      return std::chrono::duration<double>(
        command_timestamp - camera_timestamp).count();
    }

    [[nodiscard]] double total() const noexcept
    {
      return beforeFire() + fire_delay;
    }
  };

struct Ballistic {
  double yaw{0.0};
  double pitch{0.0};
  double fly_time{0.0};
  bool valid{false};
};

// 有效受击面独立于 PnP 灯条尺寸；默认沿用现有火控中心 60% 区域。
enum class InPlateAimMode : std::uint8_t { Center, FixedDiagonal, Independent };

struct InPlateAimConfig {
  bool enable{false};
  InPlateAimMode mode{InPlateAimMode::Independent};
  double small_width{0.135};
  double large_width{0.230};
  double height{0.125};
  double hit_margin_ratio{0.6};
  double edge_margin{0.005};
  double uncertainty_sigma{2.0};
  double lookahead{0.08};
  double max_speed{0.20};
  double max_acceleration{1.5};
  double response_time{0.12};
  double direction_enter_rate{0.15};
  double direction_exit_rate{0.05};
  int warmup_frames{3};
  bool outpost_common_height{true}; // 三块板存在共有高度时优先保持同一高度
};

// 查询使用不可变快照；只有本帧选板成功后才提交一次。
struct InPlateAimState {
  int robot_id{-1};
  int armor_id{-1};
  int stable_frames{0};
  int direction{0};
  TimePoint timestamp{}; // 发射/控制时刻，不能用随弹道迭代变化的命中时刻推进
  Eigen::Vector2d offset{Eigen::Vector2d::Zero()};
  Eigen::Vector2d velocity{Eigen::Vector2d::Zero()};
  Eigen::Vector3d position_world{Eigen::Vector3d::Zero()}; // 上次已选瞄点，用于保持换板高度
  TimePoint gun_timestamp{}; // 实际反馈时刻，与发射延迟分开
  double gun_yaw{0.0};
  double gun_yaw_rate{0.0};
};

struct InPlateAimPoint {
  Eigen::Vector3d position_world{Eigen::Vector3d::Zero()};
  Eigen::Vector2d offset{Eigen::Vector2d::Zero()};
  Eigen::Vector2d velocity{Eigen::Vector2d::Zero()};
  Eigen::Vector2d limits{Eigen::Vector2d::Zero()};
  int direction{0};
  int next_armor_id{-1};
  bool using_common_height{false};
  bool valid{false};
};

// 非 MPC 模式的瞄准参考。
struct AimReference {
  int target_id{-1};
  int armor_id{-1};
  TimePoint impact_time{};
  bool tracked{false};

  Eigen::Vector3d aim_point_barrel{Eigen::Vector3d::Zero()};
  Eigen::Vector3d aim_point_world{Eigen::Vector3d::Zero()};

  double yaw{0.0};
  double pitch{0.0};
  double yaw_rate{0.0};
  double pitch_rate{0.0};
  double yaw_acceleration{0.0};
  double pitch_acceleration{0.0};

  double fly_time{0.0};
  // 实体板心及完整板面姿态，始终不随瞄点偏移。
  Eigen::Vector3d armor_center_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d armor_center_barrel{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d armor_rotation_world{Eigen::Matrix3d::Identity()};
  InPlateAimPoint in_plate_aim;
  InPlateAimState aim_query_state;
  TimePoint fire_time{};
  double bullet_speed{0.0};
  double gravity{9.80665};
  double drag_coefficient{0.0};
  // 有效受击面半尺寸。瞄点运动另扣 edge_margin，不能再次缩小这里的边界。
  double hit_half_width{0.0};
  double hit_half_height{0.0};
  bool armor_geometry_valid{false};
};

// MPC 输出的单个可执行控制点。
struct AimSample {
  TimePoint execute_time{};

  double yaw{0.0};
  double pitch{0.0};
  double yaw_rate{0.0};
  double pitch_rate{0.0};
  double yaw_acceleration{0.0};
  double pitch_acceleration{0.0};
  double yaw_jerk{0.0};
  double pitch_jerk{0.0};
};

struct PlanningDiagnostics {
  int iteration_count{0};       // 最终候选的固定点迭代次数
  double fly_time_error{0.0};   // 最终相邻飞行时间误差，s
  double position_error{0.0};   // 最终相邻装甲板位置误差，m
  double angle_error{0.0};      // 最终相邻 yaw/pitch 合成误差，rad
  bool converged{false};        // 是否满足配置中的收敛条件
};

// AimReference 作为基类，使现有 L5 无需修改即可继续访问 plan.yaw/pitch。
struct AimPlan : AimReference {
  TimePoint generated_at{};

  // using_MPC=true 时非空，并按 execute_time 排列；首项是当前控制点。
  std::vector<AimSample> samples;

  bool using_MPC{false};       // false：直接参考；true：已生成 MPC samples
  ArmorTrackingPhase tracked_phase{ArmorTrackingPhase::Unlocked};
  // 将开火门控的两个来源分开保存，便于区分“未锁稳”和“已出窗”。
  bool tracked_ready{false};   // 选板状态稳定且本帧观测新鲜
  bool within_firing_window{false}; // 预计命中时装甲板位于正式射击窗口
  bool fire_permitted{false};  // 稳定跟踪、位于射击窗口内且弹道有效
  bool valid{false};           // 规划结果是否有效
};

struct PlannerConfig {
  InPlateAimConfig in_plate_aim;
  int max_iterations{20}; // 单块装甲板最大固定点迭代次数
  std::chrono::microseconds fly_time_tolerance{200}; // 飞行时间收敛阈值
  double position_tolerance{0.005};  // meter
  double angle_tolerance{0.0};       // rad，0 表示暂不启用

  double gravity{9.80665}; // 重力加速度绝对值，单位 m/s^2
  // 弹道模型选择。关闭时使用真空弹道；开启时使用 a_drag=-k*v。
  bool enable_air_resistance{false};
  // 有效线性阻力系数 k，单位 s^-1，需要通过实弹落点标定。
  double linear_drag_coefficient{0.0};

  // 切换或首次锁板时，yaw/pitch 分轴判断是否已经到位，单位 degree。
  double switch_yaw_dead_zone{5.0};
  double switch_pitch_dead_zone{5.0};
  // Tracking 状态下，仅当其他装甲板的评分至少高出该值时才主动切换。
  double score_switch_threshold{0.10};
  // 同一候选的评分优势连续满足阈值达到该帧数后才主动切换。
  int score_switch_stable_frames{3};
  double rotation_rate_dead_zone{0.05}; // rad/s，低于该值按近似静止处理
  int lock_stable_frames{3}; // 新装甲板连续稳定的确认帧数
  // Q_aim_cost 的平滑归一化区间，使用当前云台姿态到候选弹道角的合成角差。
  double aim_cost_good_angle{5.0};  // degree，小于该角度时评分为 1
  double aim_cost_bad_angle{30.0};  // degree，大于该角度时评分为 0

  double normal_enter_angle{60.0};   // degree
  double normal_leave_angle{20.0};   // degree
  double outpost_enter_angle{70.0};  // degree
  double outpost_leave_angle{30.0};  // degree
  // 按装甲板相对视线角速度分段线性缩小选板和开火窗口。
  bool enable_dynamic_windows{false};
  double window_shrink_start_speed{2.0}; // rad/s，低于该值不再放宽
  double window_shrink_end_speed{8.0};   // rad/s，高于该值不再收窄
  // 选板窗口从原始窗口线性缩到该比例。
  double selection_min_window_scale{0.65};
  // 开火窗口从低速比例线性缩到高速比例，且始终窄于选板窗口。
  double firing_max_window_scale{0.80};
  double firing_min_window_scale{0.40};
  // 当前已锁定板的保持窗口在动态选板窗口两侧各放宽的角度。
  double selection_hold_margin{8.0}; // degree
  int max_lost_frames{3};

  double yaw_angle_weight{9000000.0};
  double yaw_velocity_weight{0.0};
  double yaw_acceleration_weight{1.0};
  double pitch_angle_weight{9000000.0};
  double pitch_velocity_weight{0.0};
  double pitch_acceleration_weight{1.0};
  double min_yaw_acceleration{-50.0};
  double max_yaw_acceleration{50.0};
  double min_pitch_acceleration{-100.0};
  double max_pitch_acceleration{100.0};

  // TinyMPC 使用 README 中的二阶模型和 100 点、10 ms 的固定窗口。
  // 关闭时 Planner 保持原有的单点直瞄输出。
  bool enable_mpc{true};
  int mpc_max_iterations{10};
  double mpc_admm_rho{1.0}; // README 指定的 ADMM 罚参数
  double mpc_primal_tolerance{1e-3};
  double mpc_dual_tolerance{1e-3};

};

}  // namespace L4Planning
