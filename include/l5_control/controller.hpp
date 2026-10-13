#pragma once

#include "l3_estimation/armor/eskf_target.hpp"
#include "l4_planning/types.hpp"
#include "l5_control/fire_decision.hpp"
#include "l5_control/serial_command.hpp"

#include <Eigen/Geometry>

#include <optional>

namespace L5Control {

/**
 * @brief L5 的单一运行时入口
 *
 * 组装 FireInput、执行开火判定，并生成串口命令。内部保留上一条命令，用于命令
 * 跳变检查和安全保持。
 */
class Controller {
public:
  /**
   * @brief 用给定开火参数构造
   * @param fire_config 开火判定参数
   */
  explicit Controller(FireConfig fire_config = {}) noexcept;

  /**
   * @brief 处理一帧：判定开火并生成串口命令
   * @param target L3 的目标快照，只取类别
   * @param track_state 跟踪状态
   * @param plan 本帧规划
   * @param actual_pose L1 回传的枪管实际姿态；yaw/pitch 分解由 L5 完成
   * @return 规划成功时为新命令；失败时为 safeHold() 的保持命令；从未下发过命令
   *         且规划失败时为空
   */
  [[nodiscard]] std::optional<SerialCommand> update(
    const std::optional<L3Estimation::EskfTarget>& target,
    L3Estimation::TrackState track_state,
    const L4Planning::Plan& plan,
    const std::optional<Eigen::Quaterniond>& actual_pose);

  /**
   * @brief 最近一次 update() 的开火判定
   *
   * 调试叠加层要显示拒绝原因，命令本身看不出"为什么不开火"。
   *
   * @return 开火判定
   */
  const FireDecision& lastDecision() const noexcept
  {
    return last_decision_;
  }

  /**
   * @brief 保持最后 yaw/pitch 并强制关火
   * @return 保持命令；尚未有过命令时返回空
   */
  [[nodiscard]] std::optional<SerialCommand> safeHold() const;

  /**
   * @brief 由规划和开火判定组出命令；离线回放直接拿 FireDecision 组命令时也用它
   * @param plan 本帧规划，提供 yaw/pitch
   * @param decision 开火判定，提供 shoot
   * @return 串口命令；规划失败时返回空
   */
  [[nodiscard]] static std::optional<SerialCommand> makeCommand(
    const L4Planning::Plan& plan, const FireDecision& decision);

private:
  /// 开火判定
  FireDecider fire_decider_;

  /// 上一条下发的命令
  std::optional<SerialCommand> last_command_;

  /// 最近一次开火判定
  FireDecision last_decision_;
};

}  // namespace L5Control
