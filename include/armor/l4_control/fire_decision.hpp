#pragma once

#include "l2_estimation/armor/eskf_target.hpp"
#include "l3_planning/types.hpp"
#include "l4_control/reject_reason.hpp"

#include <numbers>
#include <optional>
#include <vector>

namespace L5Control {

struct FireConfig {
  // 总开火开关；关闭时仍计算 fire_feasible，便于无弹调试。
  bool shoot_enable{false};
  // 可独立关闭这两项禁火；误差及跟踪状态仍继续记录。
  bool stop_fire_on_temp_lost{true};
  bool stop_fire_on_aim_error{true};

  // 装甲板尺寸单位为米，用于把可命中区域换算为角度容差。
  double armor_width_small{0.135};
  double armor_width_big{0.230};
  double armor_height{0.125};
  double hit_margin_ratio{0.6};  // 将板面尺寸缩放为瞄点周围的容差，范围 (0, 1]
  // 关于当前瞄准指令的对称角度容差下限。
  double min_yaw_tolerance{0.5 * std::numbers::pi / 180.0};
  double min_pitch_tolerance{0.5 * std::numbers::pi / 180.0};
};

struct FireInput {
  std::optional<L3Estimation::EskfTarget> target;
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  L4Planning::AimPlan plan;

  // 必须是 L1 回传的实际云台角，而不是上一帧命令值。
  double actual_yaw{0.0};
  double actual_pitch{0.0};

  // 当前规划 yaw 相对上一条下发命令是否超过连续性阈值。
  bool command_jump{false};

  // false 表示实测弹速和配置回退弹速都不可用。
  // 使用有效配置回退值时应传 true，允许继续开火判定。
  bool bullet_speed_valid{true};
};

struct AimTolerance {
  double yaw{0.0};    // rad
  double pitch{0.0};  // rad
  bool valid{false};
};

struct FireDecision {
  // fire_feasible 不考虑 shoot_enable；shoot 是最终下发值。
  bool fire_feasible{false};
  bool shoot{false};
  AimTolerance tolerance;
  double yaw_error{0.0};
  double pitch_error{0.0};
  bool aim_error_within_tolerance{false};
  // 以下板面交点结果仅用于诊断，不参与开火误差判定。
  Eigen::Vector2d predicted_hit_uv{Eigen::Vector2d::Zero()};
  double predicted_fly_time{0.0};
  bool hit_inside_plate{false};
  std::vector<RejectReason> reasons;
};

class FireDecider {
public:
  explicit FireDecider(FireConfig config = {}) noexcept;

  [[nodiscard]] FireDecision decide(const FireInput& input) const;
  // 根据瞄点距离和装甲板尺寸、朝向，计算关于瞄准点的对称 yaw/pitch 容差。
  // 命中窗口。要板型定宽高，还要类别定后仰角——前哨站的板反着倾。
  AimTolerance tolerance(
    const L4Planning::AimPlan& plan,
    const std::optional<Eigen::Vector4d>& armor_pose,
    L3Estimation::ArmorType type,
    L3Estimation::ArmorName name) const noexcept;
  const FireConfig& config() const noexcept { return config_; }

private:
  FireConfig config_;
};

}  // namespace L5Control
