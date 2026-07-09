#pragma once

#include <string>

namespace L5Control {

enum class RejectReason {
  None,
  NoTarget,
  OutOfRange,
  HeatLimit,
  Unstable
};

std::string toString(RejectReason reason);

}  // namespace L5Control
