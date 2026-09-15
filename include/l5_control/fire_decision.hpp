#pragma once

#include "l3_estimation/armor/target_estimator.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/reject_reason.hpp"

#include <numbers>
#include <optional>
#include <vector>

namespace L5Control {

struct FireConfig {
  // 总开火开关；关闭时仍计算 fire_feasible，便于无弹调试。
  bool shoot_enable{false};

  // 装甲板尺寸单位为米，用于把可命中区域换算为角度容差。
  double armor_width_small{0.135};
  double armor_width_big{0.230};
  double armor_height{0.125};
  double hit_margin_ratio{0.6};  // 只使用板面中心区域，范围 (0, 1]
  double min_yaw_tolerance{0.5 * std::numbers::pi / 180.0};
  double min_pitch_tolerance{0.5 * std::numbers::pi / 180.0};
};

struct FireInput {
  std::optional<L3Estimation::TrackedTarget> target;
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};
  L4Planning::Plan plan;

  // 必须是 L1 回传的实际云台角，而不是上一帧命令值。
  double actual_yaw{0.0};
  double actual_pitch{0.0};

  // 当前规划 yaw 相对上一条下发命令是否超过连续性阈值。
  bool command_jump{false};
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
  std::vector<RejectReason> reasons;
};

class FireDecider {
public:
  explicit FireDecider(FireConfig config = {}) noexcept;

  [[nodiscard]] FireDecision decide(const FireInput& input) const;
  // 根据实体板尺寸、距离和朝向计算本帧 yaw/pitch 容差。
  // 命中窗口。要板型定宽高，还要类别定后仰角——前哨站的板反着倾。
  AimTolerance tolerance(
    const L4Planning::Plan& plan, L3Estimation::ArmorType type,
    L3Estimation::ArmorName name) const noexcept;
  const FireConfig& config() const noexcept { return config_; }

private:
  FireConfig config_;
};

}  // namespace L5Control
