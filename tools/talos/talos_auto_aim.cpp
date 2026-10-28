#include "runtime/auto_aim_runtime.hpp"

#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

namespace {
volatile std::sig_atomic_t stop_requested = 0;
void requestStop(int) { stop_requested = 1; }
} // namespace

int main(int argc, char** argv)
{
  // 兼容不同 xmake 版本：有的会剥掉 `--`，有的会原样传给程序。
  const std::string config = argc > 2 && std::string{argv[1]} == "--"
    ? argv[2]
    : (argc > 1 ? argv[1] : "config/talos.yaml");
  std::signal(SIGINT, requestStop);
  std::signal(SIGTERM, requestStop);
  try {
    runtime::AutoAimRuntime runtime(config);
    std::jthread stopper([&](std::stop_token stop_token) {
      while (!stop_token.stop_requested()) {
        if (stop_requested != 0) {
          runtime.stop();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{20});
      }
    });
    try {
      runtime.run();
    } catch (...) {
      stopper.request_stop();
      throw;
    }
    stopper.request_stop();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "talos_auto_aim: " << error.what() << '\n';
    return 1;
  }
}
