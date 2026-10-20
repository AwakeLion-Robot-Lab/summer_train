#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace runtime {

class RuntimeIo;

class AutoAimRuntime {
public:
  explicit AutoAimRuntime(const std::string& config_path);
  void run();
  void stop();

private:
  std::atomic<bool> running_{false};
  mutable std::mutex io_mutex_;
  std::shared_ptr<RuntimeIo> active_io_;
  std::string config_path_;
};

}  // namespace runtime
