#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace L1Sensor {
class Camera;
}

namespace runtime {

class AutoAimRuntime {
public:
  explicit AutoAimRuntime(const std::string& config_path);
  void run();
  void stop();

private:
  std::atomic<bool> running_{false};
  mutable std::mutex camera_mutex_;
  std::shared_ptr<L1Sensor::Camera> active_camera_;
  std::string config_path_;
};

}  // namespace runtime
