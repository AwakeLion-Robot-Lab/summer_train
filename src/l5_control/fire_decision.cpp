#include "l5_control/fire_decision.hpp"

#include "l6_telemetry/math.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace L5Control {

FireDecider::FireDecider(FireConfig config) noexcept
: config_(std::move(config))
{
}

FireDecision FireDecider::decide(const FireInput& input) const
{
  FireDecision decision;
  const auto& plan = input.plan;

  const auto reject = [&decision](RejectReason reason) {
    decision.reasons.push_back(reason);
  };

  // 不短路：一次记录本帧所有拒绝原因，便于回放直接定位多个同时存在的问题。
  if (!config_.shoot_enable) {
    reject(RejectReason::ShootDisabled);
  }
  if (input.command_jump) {
    reject(RejectReason::CommandJump);
  }

  if (!input.target) {
    reject(RejectReason::NoTarget);
  } else {
    switch (input.track_state) {
      case L3Estimation::TrackState::Lost:
      case L3Estimation::TrackState::Detecting:
        reject(RejectReason::NotTracking);
        break;
      case L3Estimation::TrackState::TempLost:
        // 短时丢失时状态全靠外推，位置误差随丢失时长增长，不允许开火。
        reject(RejectReason::TempLost);
        break;
      case L3Estimation::TrackState::Tracking:
        break;
    }
  }

  switch (plan.error) {
    case L4Planning::PlanError::None:
      break;
    case L4Planning::PlanError::NoTarget:
      // 没目标时上面已经记过。
      if (input.target) {
        reject(RejectReason::NoTarget);
      }
      break;
    case L4Planning::PlanError::OutOfWindow:
      reject(RejectReason::OutOfWindow);
      break;
    case L4Planning::PlanError::BallisticFailed:
      reject(RejectReason::BallisticFailed);
      break;
  }

  // 瞄准误差只对成功的规划有意义：失败时没有要命中的板，原因上面已经记了。
  // plan.aim 的有限性由 Planner 保证，这里只验 MCU 回传的实际角。
  if (plan.valid()) {
    if (!std::isfinite(input.actual_yaw) || !std::isfinite(input.actual_pitch)) {
      reject(RejectReason::NoPose);
    } else {
      // 命中判据：实际枪管指向与规划角之差必须落在实体板的角度投影内。
      decision.tolerance = tolerance(
        plan, input.target.value_or(L3Estimation::ArmorName::Unknown));
      decision.yaw_error =
        std::abs(L6Telemetry::limit_rad(plan.aim.yaw - input.actual_yaw));
      decision.pitch_error =
        std::abs(L6Telemetry::limit_rad(plan.aim.pitch - input.actual_pitch));
      if (!decision.tolerance.valid ||
          decision.yaw_error > decision.tolerance.yaw ||
          decision.pitch_error > decision.tolerance.pitch) {
        reject(RejectReason::AimError);
      }
    }
  }

  // ShootDisabled 只控制最终输出，不改变理论开火窗口；因此关闭总开关时仍能
  // 通过 fire_feasible 观察判定时序。
  decision.fire_feasible = std::all_of(
    decision.reasons.begin(), decision.reasons.end(),
    [](RejectReason reason) { return reason == RejectReason::ShootDisabled; });
  decision.shoot = decision.fire_feasible && config_.shoot_enable;
  return decision;
}

AimTolerance FireDecider::tolerance(
  const L4Planning::Plan& plan, L3Estimation::ArmorName name) const noexcept
{
  AimTolerance result;
  if (!plan.fire.has_value() || plan.fire->armor_id < 0) {
    return result;
  }

  const Eigen::Vector3d point = plan.fire->point();
  const double horizontal = std::hypot(point.x(), point.y());
  const double slant = std::hypot(horizontal, point.z());
  if (!std::isfinite(horizontal) || horizontal < 1e-3 || !std::isfinite(slant)) {
    return result;
  }

  // 未知类别按小板算，窗口只会更紧。
  const double width = L3Estimation::armorTypeOf(name) == L3Estimation::ArmorType::Big
                         ? config_.armor_width_big
                         : config_.armor_width_small;

  // 板面斜对枪口时，水平可命中宽度按 cos(facing_angle) 收缩。
  // 正对时取完整宽度，接近侧对时逐渐收紧到最小 yaw 容差。
  const double facing = std::abs(std::cos(plan.fire->facingAngle()));
  const double half_width = 0.5 * width * config_.hit_margin_ratio * facing;

  // 竖直方向同理，只是收缩量由两个角相加决定：装甲板本身后仰 α，视线仰角 β，
  // 可见高度是 h·|cos(α + β)|。板顶后仰、又从下往上看时两者叠加，可命中的
  // 竖直窗口比板高小得多；俯角恰好抵消后仰时（α + β = 0）才看到完整板高。
  //
  // 用视线仰角而不是枪管 pitch：枪管 pitch 含弹道抬升，不是看过去的方向，
  // 而这里要的是"从射手位置看这块板有多高"。
  const double line_of_sight_pitch = std::atan2(point.z(), horizontal);
  const double tilt =
    std::abs(std::cos(L3Estimation::armorPitchOf(name) + line_of_sight_pitch));
  const double half_height =
    0.5 * config_.armor_height * config_.hit_margin_ratio * tilt;

  // yaw 是水平角，用水平距离；pitch 是竖直角，用斜距。
  result.yaw = std::max(std::atan2(half_width, horizontal), config_.min_yaw_tolerance);
  result.pitch = std::max(std::atan2(half_height, slant), config_.min_pitch_tolerance);
  result.valid = std::isfinite(result.yaw) && std::isfinite(result.pitch);
  return result;
}

}  // namespace L5Control
