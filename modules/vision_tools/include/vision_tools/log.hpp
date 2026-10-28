#pragma once

#include <iostream>
#include <mutex>

namespace vision_tools {

template <typename... Args>
void logMessage(const char* level, const Args&... args)
{
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::cerr << "[vision_tools][" << level << ']';
  ((std::cerr << ' ' << args), ...);
  std::cerr << '\n';
}

template <typename... Args>
void logInfo(const Args&... args) { logMessage("INFO", args...); }

template <typename... Args>
void logWarn(const Args&... args) { logMessage("WARN", args...); }

template <typename... Args>
void logError(const Args&... args) { logMessage("ERROR", args...); }

}  // namespace vision_tools
