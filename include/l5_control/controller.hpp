#pragma once

#include "l3_estimation/armor/eskf_target.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"
#include "l5_control/serial_command.hpp"

#include <Eigen/Geometry>

#include <optional>

namespace L5Control {

// L5 的单一运行时入口：组装 FireInput、执行开火判定，并生成
// 串口命令。内部保留上一条命令，用于命令跳变检查和安全保持。
class Controller {
public:
  explicit Controller(FireConfig fire_config = {}) noexcept;

  // actual_pose 是 L1 回传的枪管实际姿态；yaw/pitch 分解由 L5 完成。
  [[nodiscard]] std::optional<SerialCommand> update(
    const std::optional<L3Estimation::EskfTarget>& target,
    L3Estimation::TrackState track_state,
    const L4Planning::Plan& plan,
    const std::optional<Eigen::Quaterniond>& actual_pose);

  // 最近一次 update() 的开火判定。调试叠加层要显示拒绝原因，命令本身
  // 看不出"为什么不开火"。
  const FireDecision& lastDecision() const noexcept
  {
    return last_decision_;
  }

  // 保持最后 yaw/pitch 并强制关火；尚未有过命令时返回空。
  [[nodiscard]] std::optional<SerialCommand> safeHold() const;
  // 规划失败时返回空。离线回放直接拿 FireDecision 组命令时也用它。
  [[nodiscard]] static std::optional<SerialCommand> makeCommand(
    const L4Planning::Plan& plan, const FireDecision& decision);

private:
  FireDecider fire_decider_;
  std::optional<SerialCommand> last_command_;
  FireDecision last_decision_;
};

}  // namespace L5Control
