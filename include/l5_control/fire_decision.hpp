#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/tracked_target.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/reject_reason.hpp"

#include <chrono>
#include <numbers>
#include <optional>
#include <vector>

namespace L5Control {

struct FireConfig {
  // 第一阶段必须保持 false；完成验收后由实车配置显式修改。
  bool shoot_enable{false};

  std::optional<double> bullet_diameter;  // meter; 17 mm projectile = 0.017
  std::optional<double> min_bullet_speed;  // meter per second
  std::optional<double> max_bullet_speed;  // meter per second
  std::optional<double> heat_limit;

  std::optional<double> min_yaw;
  std::optional<double> max_yaw;
  std::optional<double> min_pitch;
  std::optional<double> max_pitch;

  std::chrono::milliseconds max_robot_state_age{50};
  std::chrono::milliseconds max_gimbal_pose_age{20};

  // ---- 命中判据 ----
  //
  // 角度容差由**装甲板的物理尺寸在命中距离上张开的角度**决定，而不是一个固定角。
  //
  // 宽度与 L3 的 ArmorConfig 一致（灯条中心间距 ≈ 板宽）。
  // **高度不能照抄 ArmorConfig::height**：那里的 0.056 是灯条的竖直跨度，是 PnP
  // 的几何量；能挨枪的板面高度是 0.125 左右。用 0.056 会把竖直容差压到实际的
  // 一半以下，绝大多数距离上直接被下限接管，等于 pitch 只剩一个固定角。
  double armor_width_small{0.135};
  double armor_width_big{0.230};
  double armor_height{0.125};

  // 收缩系数：只有打在板中心附近才算数。取整块板会把擦边命中也算进来，而擦边
  // 命中在真实系统里往往因为弹丸散布和估计误差而脱靶。
  double hit_margin_ratio{0.6};

  // 角度容差下限。距离一远物理张角就趋近 0，没有下限的话远距离永远开不了火。
  double min_yaw_tolerance{0.5 * std::numbers::pi / 180.0};
  double min_pitch_tolerance{0.5 * std::numbers::pi / 180.0};

  [[nodiscard]] bool parametersReady() const noexcept
  {
    return bullet_diameter.has_value() && min_bullet_speed.has_value() &&
           max_bullet_speed.has_value() && heat_limit.has_value() &&
           min_yaw.has_value() && max_yaw.has_value() &&
           min_pitch.has_value() && max_pitch.has_value();
  }
};

struct FireInput {
  std::optional<L3Estimation::TrackedTarget> target;
  // 跟踪状态由 Tracker 持有而不是挂在目标上（TrackedTarget 只是滤波器状态的
  // 副本），所以这里单独传入 Tracker::state()。TempLost 时目标仍然有值，全靠
  // 外推，位置误差随丢失时长增长，因此火控必须能区分它和 Tracking。
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  L4Planning::Plan plan;
  L1Sensor::RobotState robot_state;

  std::chrono::steady_clock::time_point now{};

  // 云台**实际**指向，来自 L1 回传，不是上一帧下发的命令值：电机实际走的路径
  // 可能偏离规划路径，拿命令值和命令值比永远是 0，等于没有判据。
  double actual_yaw{0.0};
  double actual_pitch{0.0};

  bool calibration_ready{false};
  bool serial_fresh{false};
  bool gimbal_pose_fresh{false};
  bool armor_switching{false};
  bool command_jump{false};
};

// 装甲板在命中距离上张开的角度，即允许的瞄准误差。
struct AimTolerance {
  double yaw{0.0};    // radian
  double pitch{0.0};  // radian
  bool valid{false};
};

struct FireDecision {
  // fire_feasible 记录理论窗口；shoot 是考虑 shoot_enable 后的实际下发值。
  bool fire_feasible{false};
  bool shoot{false};

  // 本帧实际用到的容差和误差，便于回放时判断"差在哪一轴、差多少"。
  AimTolerance tolerance;
  double yaw_error{0.0};
  double pitch_error{0.0};

  std::vector<RejectReason> reasons;
};

// 火控判定。无状态，配置在构造时固定。
//
// 两条要点：
//  1. **yaw 和 pitch 都判**，俯仰没跟上同样不许开火。
//  2. **拒绝原因全部列出，不短路**：实车上"为什么不开火"如果只报第一条，就得
//     靠反复复现一条条剥，而每复现一次就是一次上场。
class FireDecider {
public:
  explicit FireDecider(FireConfig config = {}) noexcept : config_(std::move(config)) {}

  [[nodiscard]] FireDecision decide(const FireInput& input) const;

  // 把装甲板尺寸换算成该命中点上的角度容差。plan 必须已经填好
  // fire_armor_point，否则返回 valid = false。
  [[nodiscard]] AimTolerance tolerance(
    const L4Planning::Plan& plan, L3Estimation::ArmorType type) const noexcept;

  [[nodiscard]] const FireConfig& config() const noexcept { return config_; }

private:
  FireConfig config_;
};

}  // namespace L5Control
