#pragma once

#include "l3_estimation/target_estimator.hpp"
#include "l3_estimation/types.hpp"
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
  Controller() noexcept;
  Controller(FireConfig fire_config, double command_jump_threshold) noexcept;

  // actual_pose 是 L1 回传的枪管实际姿态；yaw/pitch 分解由 L5 完成。
  [[nodiscard]] std::optional<SerialCommand> update(
    const std::optional<L3Estimation::TrackedTarget>& target,
    L3Estimation::TrackState track_state,
    const L4Planning::Plan& plan,
    const std::optional<Eigen::Quaterniond>& actual_pose);

  // 保持最后 yaw/pitch 并强制关火；尚未有过命令时返回空。
  [[nodiscard]] std::optional<SerialCommand> safeHold() const;
  // 保留低层组装函数，供离线回放直接检查 FireDecision。
  [[nodiscard]] std::optional<SerialCommand> makeCommand(
    const L4Planning::Plan& plan, const FireDecision& decision) const;

private:
  FireDecider fire_decider_;
  double command_jump_threshold_{0.0};
  std::optional<SerialCommand> last_command_;
};

}  // namespace L5Control
