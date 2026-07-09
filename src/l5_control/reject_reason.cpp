#include "l5_control/reject_reason.hpp"

namespace L5Control {

std::string toString(RejectReason reason)
{
  switch (reason) {
    case RejectReason::None:
      return "none";
    case RejectReason::NoTarget:
      return "no_target";
    case RejectReason::OutOfRange:
      return "out_of_range";
    case RejectReason::HeatLimit:
      return "heat_limit";
    case RejectReason::Unstable:
      return "unstable";
    default:
      return "unknown";
  }
}

}  // namespace L5Control
