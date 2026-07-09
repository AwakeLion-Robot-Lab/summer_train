#include "l6_telemetry/logger.hpp"

#include "aw_logger/aw_logger.hpp"

#ifdef ERROR
#undef ERROR
#endif

#include <filesystem>
#include <memory>

namespace L6Telemetry {
namespace {

constexpr const char* kRootLoggerName = "AW_Vision";

}  // namespace

void initLogger()
{
  std::filesystem::create_directories("logs");

  auto logger = aw_logger::getLogger(kRootLoggerName);

  auto console = std::make_shared<aw_logger::ConsoleAppender>();
  console->setPattern("%t [%p] %f:%l %m");

  auto file = std::make_shared<aw_logger::FileAppender>("logs/logger_l6_telemetry.log");
  file->setPattern("%t [%p] %m");
  file->setMaxFileSize(1024 * 1024);
  file->setMaxBackupNum(3);

  logger->setAppenders(console, file);
  logger->setThresholdLevel(aw_logger::LogLevel::level::DEBUG);
}

void flushLogger()
{
  aw_logger::getLogger(kRootLoggerName)->flush();
}

void logDebugRaw(const std::string& msg)
{
  AW_LOG_DEBUG(aw_logger::getLogger(kRootLoggerName), msg);
}

void logInfoRaw(const std::string& msg)
{
  AW_LOG_INFO(aw_logger::getLogger(kRootLoggerName), msg);
}

void logWarnRaw(const std::string& msg)
{
  AW_LOG_WARN(aw_logger::getLogger(kRootLoggerName), msg);
}

void logErrorRaw(const std::string& msg)
{
  AW_LOG_ERROR(aw_logger::getLogger(kRootLoggerName), msg);
}

}  // namespace L6Telemetry
