#pragma once

#include "l4_planning/planner.hpp"
#include "l5_control/fire_decision.hpp"
#include "l5_control/serial_command.hpp"

#include <optional>

namespace L5Control {

// L4 的 Plan 到串口命令的最后一步。这里只做搬运和拒发判断，不做任何角度
// 修正：偏置、限幅、坐标系转换要么属于标定（L1），要么属于规划（L4），
// 混在这里会让"发出去的角"和"规划的角"对不上，实车上极难查。
class Controller {
public:
  // 规划无效时返回 nullopt，调用方不要下发。
  //
  // 不返回一条填零的命令：yaw = pitch = 0 是 world 系里一个具体方向，下位机
  // 会当成正常指令去跟，云台会猛甩。丢失目标时什么都不发、让下位机保持原状
  // 才是安全行为。
  [[nodiscard]] std::optional<SerialCommand> makeCommand(
    const L4Planning::Plan& plan, const FireDecision& decision) const;
};

}  // namespace L5Control
