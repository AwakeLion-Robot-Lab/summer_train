#pragma once

#include <source_location>
#include <sstream>
#include <string>
#include <utility>

namespace L6Telemetry {

enum class LogLevel { Debug, Info, Warn, Error };

void initLogger();
void flushLogger();

// 只给下面的宏用。位置由宏在调用处取好传进来：aw_logger 的 AW_LOG_* 宏也是
// 在展开处调 std::source_location::current()，包进函数里再调，file:line 就永远
// 指向 logger.cpp 自己。宏不直接展开成 AW_LOG_*，是因为 aw_logger 只有头文件，
// 每个打日志的 TU 都 include 它要多编约 5 秒。
void writeLog(LogLevel level, const std::source_location& where, std::string message);

// 参数用空格拼接，不是格式串：LOG_INFO("speed", v, "m/s") -> "speed 23 m/s"。
template <typename... Args>
std::string joinLog(Args&&... args)
{
  std::ostringstream oss;
  bool first = true;
  ((oss << (std::exchange(first, false) ? "" : " ") << std::forward<Args>(args)), ...);
  return oss.str();
}

}  // namespace L6Telemetry

#define NV_LOG_AT(level, ...)                                      \
  ::L6Telemetry::writeLog(                                         \
    ::L6Telemetry::LogLevel::level, std::source_location::current(), \
    ::L6Telemetry::joinLog(__VA_ARGS__))

#define LOG_DEBUG(...) NV_LOG_AT(Debug, __VA_ARGS__)
#define LOG_INFO(...) NV_LOG_AT(Info, __VA_ARGS__)
#define LOG_WARN(...) NV_LOG_AT(Warn, __VA_ARGS__)
#define LOG_ERROR(...) NV_LOG_AT(Error, __VA_ARGS__)
