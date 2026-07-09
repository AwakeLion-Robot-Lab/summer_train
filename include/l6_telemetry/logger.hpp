#pragma once

#include <sstream>
#include <string>
#include <string_view>

namespace L6Telemetry {

void initLogger();
void flushLogger();
void logDebugRaw(const std::string& msg);
void logInfoRaw(const std::string& msg);
void logWarnRaw(const std::string& msg);
void logErrorRaw(const std::string& msg);

inline std::string makeLogMessage(std::string_view fmt)
{
  return std::string(fmt);
}

template <typename... Args>
std::string makeLogMessage(std::string_view fmt, Args&&... args)
{
  std::ostringstream oss;
  oss << fmt;
  ((oss << ' ' << std::forward<Args>(args)), ...);
  return oss.str();
}

template <typename... Args>
void logDebug(std::string_view fmt, Args&&... args)
{
  logDebugRaw(makeLogMessage(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
void logInfo(std::string_view fmt, Args&&... args)
{
  logInfoRaw(makeLogMessage(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
void logWarn(std::string_view fmt, Args&&... args)
{
  logWarnRaw(makeLogMessage(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
void logError(std::string_view fmt, Args&&... args)
{
  logErrorRaw(makeLogMessage(fmt, std::forward<Args>(args)...));
}

}  // namespace L6Telemetry
