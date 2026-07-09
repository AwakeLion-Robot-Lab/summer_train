#include "l6_telemetry/logger.hpp"

int main()
{
  L6Telemetry::initLogger();
  L6Telemetry::logInfo("logger smoke test start");
  L6Telemetry::logWarn("logger smoke test finish");
  L6Telemetry::flushLogger();
  return 0;
}
