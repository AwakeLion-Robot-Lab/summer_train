#include "l6_telemetry/trace_scope.hpp"

#include "l6_telemetry/logger.hpp"

#include <utility>

namespace L6Telemetry {

TraceScope::TraceScope(std::string name)
  : name_(std::move(name))
{
  logDebug("enter {}", name_);
}

TraceScope::~TraceScope()
{
  logDebug("leave {}", name_);
}

}  // namespace L6Telemetry
