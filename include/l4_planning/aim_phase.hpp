#pragma once

#include "l4_planning/types.hpp"

namespace L4Planning {

// 随目标转速升高逐级切换的瞄准档位，抄自 talos 的 ArmorAimPhase。
//
//   SingleArmor     低速。盯正对枪口的那块板，允许锁定迟滞。
//   WholeCarArmor   中速。仍瞄实体板，但让选板跟着旋转走，不再强行锁定。
//   WholeCarCenter  高速。没有哪块板能在视野里停够久，改为把枪口压在旋转
//                   圆离枪口最近的那个点上，等板扫过来时开火。
//
// talos 在 WholeCarArmor 和 WholeCarCenter 之间还有一档 WholeCarPair。查过
// 它的 predict_aim_point 和 select_armor_id：两处都只对 WholeCarCenter 分支，
// WholeCarPair 与 WholeCarArmor 的瞄准行为完全一致，只是迟滞阶梯上的一级。
// 这里合并成三档，迟滞由下面的计数器本身保证。
enum class AimPhase : std::uint8_t {
  SingleArmor,
  WholeCarArmor,
  WholeCarCenter
};

// 档位切换必须比"和阈值比一下"更钝：v_yaw 是 EKF 估计量，在阈值附近会来回
// 抖，每抖一次换一次瞄准策略，云台就永远在追不同的点。
//
// 两层迟滞：
//  1. 上下行用不同阈值（施密特触发），中间留死区；
//  2. 条件连续成立 transfer_count 帧才真正换档。
class AimPhaseTracker {
public:
  explicit AimPhaseTracker(AimPhaseConfig config = {}) noexcept : config_(config) {}

  // abs_v_yaw 取绝对值后的整车角速度 (rad/s)。geometry_observed 为 false 时
  // 整车几何还没被观测约束过，强制留在 SingleArmor——此时连"另外几块板在
  // 哪"都不知道，谈不上整车瞄准。
  void update(double abs_v_yaw, bool geometry_observed) noexcept;

  void reset() noexcept
  {
    phase_ = AimPhase::SingleArmor;
    counter_ = 0;
  }

  [[nodiscard]] AimPhase phase() const noexcept { return phase_; }
  [[nodiscard]] int counter() const noexcept { return counter_; }
  [[nodiscard]] const AimPhaseConfig& config() const noexcept { return config_; }

private:
  AimPhaseConfig config_;
  AimPhase phase_{AimPhase::SingleArmor};
  // 正数表示朝上一档累积，负数表示朝下一档累积，条件不成立时立即清零。
  int counter_{0};
};

}  // namespace L4Planning
