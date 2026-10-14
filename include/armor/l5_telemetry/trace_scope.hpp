#pragma once

#include <string>

namespace L6Telemetry {

class TraceScope {
public:
  explicit TraceScope(std::string name);
  ~TraceScope();

private:
  std::string name_;
};

}  // namespace L6Telemetry
