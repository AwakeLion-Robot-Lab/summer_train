#include "l6_telemetry/logger.hpp"

#include "aw_logger/aw_logger.hpp"

#ifdef ERROR
#undef ERROR
#endif

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>

namespace L6Telemetry {
namespace {

const aw_logger::Logger::Ptr& logger()
{
  // LoggerManager 对同名总是返回同一个实例，initLogger 改的也是它，缓存起来
  // 省掉每条日志一次加锁查表。
  static const aw_logger::Logger::Ptr instance = aw_logger::getLogger("AW_Vision");
  return instance;
}

aw_logger::LogLevel::level toAw(LogLevel level)
{
  switch (level) {
    case LogLevel::Debug: return aw_logger::LogLevel::level::DEBUG;
    case LogLevel::Info: return aw_logger::LogLevel::level::INFO;
    case LogLevel::Warn: return aw_logger::LogLevel::level::WARN;
    case LogLevel::Error: return aw_logger::LogLevel::level::ERROR;
  }
  return aw_logger::LogLevel::level::ERROR;
}

}  // namespace

void initLogger()
{
  std::filesystem::create_directories("logs");

  auto console = std::make_shared<aw_logger::ConsoleAppender>();
  console->setPattern("%t [%p] %f:%l %m");

  // 文件里也带位置：车上出的问题只剩日志文件可看。
  auto file = std::make_shared<aw_logger::FileAppender>("logs/logger_l6_telemetry.log");
  file->setPattern("%t [%p] %f:%l %m");
  file->setMaxFileSize(1024 * 1024);
  file->setMaxBackupNum(3);

  logger()->setAppenders(console, file);
  logger()->setThresholdLevel(aw_logger::LogLevel::level::DEBUG);
}

void flushLogger()
{
  logger()->flush();
}

void writeLog(LogLevel level, const std::source_location& where, std::string message)
{
  const auto aw_level = toAw(level);
  if (aw_level < logger()->getThresholdLevel()) {
    return;
  }
  // 与 AW_LOG_BASE 的展开相同，只是位置用调用处传进来的。
  try {
    aw_logger::LogEventWrap(std::make_shared<aw_logger::LogEvent>(
      logger(), aw_level,
      aw_logger::LogEvent::LocalSourceLocation<std::string>(std::move(message), where)));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
  }
}

}  // namespace L6Telemetry
