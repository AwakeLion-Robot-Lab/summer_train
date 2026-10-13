#pragma once

#include "l3_estimation/armor/types.hpp"
#include "l3_estimation/types.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/reject_reason.hpp"

#include <numbers>
#include <optional>
#include <vector>

namespace L5Control {

/**
 * @brief 开火判定参数，角度单位 rad
 */
struct FireConfig {
  /// 总开火开关；关闭时仍计算 fire_feasible，便于无弹调试。
  bool shoot_enable{false};

  /// 小板宽度，米。装甲板尺寸用于把可命中区域换算为角度容差。
  double armor_width_small{0.135};

  /// 大板宽度，米
  double armor_width_big{0.230};

  /// 板高度，米
  double armor_height{0.125};

  /// 只使用板面中心区域，范围 (0, 1]
  double hit_margin_ratio{0.6};

  /// yaw 容差下限，rad
  double min_yaw_tolerance{0.5 * std::numbers::pi / 180.0};

  /// pitch 容差下限，rad
  double min_pitch_tolerance{0.5 * std::numbers::pi / 180.0};

  /// yaw 命令与上一条下发命令相差超过它（rad）时，本帧以 command_jump 拒绝开火，
  /// 命令照发。由 Controller 比对，因为只有它保留着上一条命令。
  double command_jump_threshold{10.0 * std::numbers::pi / 180.0};
};

/**
 * @brief 一帧开火判定的输入
 */
struct FireInput {
  /// 目标类别，用来定板宽和后仰角；没有目标时为空。火控不读任何滤波器状态。
  std::optional<L3Estimation::ArmorName> target;

  /// 跟踪状态，只有 Tracking 允许开火
  L3Estimation::TrackState track_state{L3Estimation::TrackState::Lost};

  /// 本帧规划
  L4Planning::Plan plan;

  /// 云台实际 yaw，rad。必须是 L1 回传的实际云台角，而不是上一帧命令值；拿不到时填 NaN。
  double actual_yaw{0.0};

  /// 云台实际 pitch，rad，同上
  double actual_pitch{0.0};

  /// 当前规划 yaw 相对上一条下发命令是否超过连续性阈值。
  bool command_jump{false};
};

/**
 * @brief 本帧的 yaw/pitch 命中窗口
 */
struct AimTolerance {
  /// yaw 半宽，rad
  double yaw{0.0};

  /// pitch 半宽，rad
  double pitch{0.0};

  /// 是否算得出来
  bool valid{false};
};

/**
 * @brief 开火判定结果
 */
struct FireDecision {
  /// 理论上可开火，不考虑 shoot_enable
  bool fire_feasible{false};

  /// 最终下发值
  bool shoot{false};

  /// 命中窗口
  AimTolerance tolerance;

  /// 规划 yaw 与实际 yaw 之差的绝对值，rad
  double yaw_error{0.0};

  /// 规划 pitch 与实际 pitch 之差的绝对值，rad
  double pitch_error{0.0};

  /// 拒绝原因，每种情况只出现一次；为空表示可开火
  std::vector<RejectReason> reasons;
};

/**
 * @brief 开火判定：跟踪状态、规划结果、瞄准误差和命令连续性的综合判断
 */
class FireDecider {
public:
  /**
   * @brief 用给定参数构造
   * @param config 开火判定参数
   */
  explicit FireDecider(FireConfig config = {}) noexcept;

  /**
   * @brief 判定本帧是否开火
   *
   * 不短路：一次记下本帧所有拒绝原因，便于回放直接定位多个同时存在的问题。
   *
   * @param input 本帧输入
   * @return 判定结果
   */
  [[nodiscard]] FireDecision decide(const FireInput& input) const;

  /**
   * @brief 根据实体板尺寸、距离和朝向计算本帧 yaw/pitch 命中窗口
   *
   * 类别定板宽，也定后仰角——前哨站的板反着倾。
   *
   * @param plan 规划结果，取目标板与命中点
   * @param name 车辆类别；未知类别按小板算，窗口只会更紧
   * @return 命中窗口；规划没有目标板或命中点距离退化时 valid 为 false
   */
  AimTolerance tolerance(
    const L4Planning::Plan& plan, L3Estimation::ArmorName name) const noexcept;

  /**
   * @brief 开火判定参数
   * @return 参数
   */
  const FireConfig& config() const noexcept { return config_; }

private:
  /// 开火判定参数
  FireConfig config_;
};

}  // namespace L5Control
