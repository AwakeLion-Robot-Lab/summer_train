#pragma once

#include "l4_planning/buff/types.hpp"
#include "l5_control/serial_command.hpp"

#include <optional>

namespace L5Control::Buff {

struct Config {
  // 源 FireController 的 Rune 分支默认自动开火；需要仅跟随时在 buff.yaml 关闭。
  bool shoot_enable{true};
};

class Controller {
public:
  explicit Controller(Config config = {}) : config_(config) {}

  [[nodiscard]] std::optional<SerialCommand> update(
    const L4Planning::Buff::Plan& plan);
  [[nodiscard]] std::optional<SerialCommand> safeHold() const;
  void reset() noexcept;

private:
  Config config_;
  std::optional<SerialCommand> last_command_;
};

}  // namespace L5Control::Buff
