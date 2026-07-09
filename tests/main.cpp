#include "l6_telemetry/logger.hpp"
#include "runtime/auto_aim_runtime.hpp"

int main()
{
  L6Telemetry::initLogger();
  runtime::AutoAimRuntime runtime("config/carmera_config.yaml");
  runtime.run();
  L6Telemetry::flushLogger();
  return 0;
}
