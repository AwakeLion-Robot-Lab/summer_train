#pragma once

#include <atomic>
#include <string>

namespace runtime {

class AutoAimRuntime {
public:
  explicit AutoAimRuntime(const std::string& config_path);
  void run();
  void stop();

private:
  std::atomic<bool> running_{false};
  std::string config_path_;
};

}  // namespace runtime
