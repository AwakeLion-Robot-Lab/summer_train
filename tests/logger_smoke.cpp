#include "l6_telemetry/logger.hpp"

int main()
{
  L6Telemetry::initLogger();
  LOG_INFO("logger smoke test start");
  LOG_WARN("logger smoke test finish");
  L6Telemetry::flushLogger();
  return 0;
}
