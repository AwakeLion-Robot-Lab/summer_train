#pragma once

#include "l1_sensor/serial/robot_state.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/planner.hpp"
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
  // 角度容差由**装甲板的物理尺寸在命中距离上张开的角度**决定，而不是一个拍脑袋
  // 的固定角。这是工作空间里所有认真做火控的项目的共识做法（talos
  // `is_on_target`、FYT `isOnTarget`、awakening `very_aimer`、jlu
  // `calculateFireThres`、rm.cv.fans 的投影半宽），差别只在细节。
  //
  // 宽度与 L3 的 ArmorConfig 一致（灯条中心间距 ≈ 板宽）。
  //
  // **高度不能照抄 ArmorConfig::height。** 那里的 0.056 是灯条的竖直跨度，
  // 是 PnP 的几何量；能挨枪的板面高度是 0.125 左右。拿 0.056 当命中判据会把
  // 竖直容差压到实际的一半以下，绝大多数距离上直接被下限接管，等于 pitch 只
  // 剩一个固定角。参考实现（talos / FYT / awakening 的 shooting_range_h）
  // 一律用 0.12。
  double armor_width_small{0.135};
  double armor_width_big{0.230};
  double armor_height{0.125};

  // 收缩系数：只有打在板中心附近才算数。取整块板会把擦边命中也算进来，而擦边
  // 命中在真实系统里往往因为弹丸散布和估计误差而脱靶。
  double hit_margin_ratio{0.6};

  // 角度容差下限。距离一远物理张角就趋近 0，没有下限的话远距离永远开不了火。
  // 0.5° 与 talos 标定的 0.008444 rad 同量级。
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
  std::optional<L3Estimation::Target> target;
  L4Planning::Plan plan;
  L1Sensor::RobotState robot_state;

  std::chrono::steady_clock::time_point now{};

  // 云台**实际**指向，来自 L1 回传，不是上一帧下发的命令值。
  //
  // 必须用实际角。jlu 在 fire_controller.hpp 里把理由写得最清楚：电机实际走的
  // 路径可能偏离规划路径，拿命令值和命令值比永远是 0，等于没有判据。sp_vision
  // 和 Climber_Vision 的 Shooter 也是拿 gimbal_pos 和**上一帧**命令比，同一个
  // 道理。
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

// 火控判定。无状态，配置在构造时固定，和 BallisticSolver / Planner 的风格一致。
//
// 三条与参考实现不同的取舍：
//
//  1. **yaw 和 pitch 都判。** sp_vision 和 Climber_Vision 的 Shooter 只判 yaw
//     （`shooter.cpp` 里没有任何 pitch 项），俯仰没跟上也照开。talos 和
//     awakening 两轴都判，这里跟后者。
//  2. **判在实体装甲板上，不是判在瞄准点上。** WholeCarCenter 档瞄的是旋转圆
//     上的代理点，拿它判等于在两块板之间的空档里照样开火。用
//     `Plan::fire_armor_id` / `fire_armor_point` 那块实体板。talos 用
//     `fire_aim_phase()` 把 WholeCarCenter 降级成 WholeCarArmor 单独算一条
//     reference trajectory 来表达同一件事。
//  3. **拒绝原因全部列出，不短路。** 和 L3 的 ArmorQuality 一样：一次看清所有
//     没开火的原因，比只看到第一条有用得多。
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

// 兼容旧调用点。新代码应当用 FireDecider::decide()，它给得出拒绝原因。
bool shouldFire(const L4Planning::AimPlan& plan);

}  // namespace L5Control
