#pragma once

#include <source_location>
#include <sstream>
#include <string>
#include <utility>

namespace L6Telemetry {

/**
 * @brief 日志级别
 */
enum class LogLevel { Debug, Info, Warn, Error };

/**
 * @brief 初始化日志：控制台与 logs/ 下的文件两路输出，进程启动时调用一次
 */
void initLogger();

/**
 * @brief 把缓冲中的日志刷到文件，进程退出前调用
 */
void flushLogger();

/**
 * @brief 写一条日志，只给下面的宏用
 *
 * 位置由宏在调用处取好传进来：aw_logger 的 AW_LOG_* 宏也是在展开处调
 * std::source_location::current()，包进函数里再调，file:line 就永远指向
 * logger.cpp 自己。宏不直接展开成 AW_LOG_*，是因为 aw_logger 只有头文件，
 * 每个打日志的 TU 都 include 它要多编约 5 秒。
 *
 * @param level 日志级别
 * @param where 调用处位置
 * @param message 已拼好的消息
 */
void writeLog(LogLevel level, const std::source_location& where, std::string message);

/**
 * @brief 把参数用空格拼成一条消息
 *
 * 参数用空格拼接，不是格式串：LOG_INFO("speed", v, "m/s") -> "speed 23 m/s"。
 *
 * @tparam Args 任意可 operator<< 的类型
 * @param args 要拼接的参数
 * @return 拼好的消息
 */
template <typename... Args>
std::string joinLog(Args&&... args)
{
  std::ostringstream oss;
  bool first = true;
  ((oss << (std::exchange(first, false) ? "" : " ") << std::forward<Args>(args)), ...);
  return oss.str();
}

}  // namespace L6Telemetry

/// 在调用处取 source_location 并写日志，见 writeLog
#define NV_LOG_AT(level, ...)                                      \
  ::L6Telemetry::writeLog(                                         \
    ::L6Telemetry::LogLevel::level, std::source_location::current(), \
    ::L6Telemetry::joinLog(__VA_ARGS__))

/// Debug 级日志，参数用空格拼接
#define LOG_DEBUG(...) NV_LOG_AT(Debug, __VA_ARGS__)

/// Info 级日志，参数用空格拼接
#define LOG_INFO(...) NV_LOG_AT(Info, __VA_ARGS__)

/// Warn 级日志，参数用空格拼接
#define LOG_WARN(...) NV_LOG_AT(Warn, __VA_ARGS__)

/// Error 级日志，参数用空格拼接
#define LOG_ERROR(...) NV_LOG_AT(Error, __VA_ARGS__)
