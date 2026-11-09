#include "runtime/single_frame_pipeline.hpp"

#include <atomic>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
}

int main() {
  try {
    using Pipeline = runtime::SingleFramePipeline<std::unique_ptr<int>>;
    std::atomic<int> produced{0};
    std::promise<void> first, second;
    {
      Pipeline pipe([&]() -> std::optional<std::unique_ptr<int>> {
        const int id = ++produced;
        if (id == 1) first.set_value();
        if (id == 2) second.set_value();
        return std::make_unique<int>(id);
      });
      require(first.get_future().wait_for(1s) == std::future_status::ready, "producer did not start");
      std::this_thread::sleep_for(20ms);
      require(produced == 1, "producer advanced before first handoff");
      std::unique_ptr<int> packet;
      require(pipe.readFor(packet, 1s) == Pipeline::ReadStatus::Value, "missing first result");
      require(*packet == 1, "incorrect first result");
      require(second.get_future().wait_for(1s) == std::future_status::ready,
              "next frame cannot overlap consumer work");
      std::this_thread::sleep_for(20ms); // Model a slow planner/display.
      require(produced == 2, "more than one frame ran ahead of consumer");
      require(*packet == 1, "producer changed consumer-owned snapshot");
      // Destructor must wake the producer even with an unconsumed full slot.
    }
    require(produced == 2, "shutdown started another frame");

    using IntPipeline = runtime::SingleFramePipeline<int>;
    {
      int id = 0;
      IntPipeline pipe([&]() -> std::optional<int> {
        if (id == 200) return std::nullopt;
        return id++;
      });
      int packet = -1;
      for (int expected = 0; expected < 200; ++expected) {
        require(pipe.readFor(packet, 1s) == IntPipeline::ReadStatus::Value && packet == expected,
                "result dropped, duplicated or reordered");
      }
      require(pipe.readFor(packet, 1s) == IntPipeline::ReadStatus::Closed, "EOF did not close");
    }
    {
      std::promise<void> release;
      auto gate = release.get_future().share();
      IntPipeline pipe([gate]() -> std::optional<int> {
        gate.wait_for(100ms); // Bounded even if the test fails.
        return 7;
      });
      int packet = -1;
      require(pipe.readFor(packet, 1ms) == IntPipeline::ReadStatus::Timeout,
              "empty handoff did not time out");
      release.set_value();
      require(pipe.readFor(packet, 1s) == IntPipeline::ReadStatus::Value && packet == 7,
              "timeout lost the next result");
    }
    {
      IntPipeline pipe([]() -> std::optional<int> { throw std::runtime_error("producer failure"); });
      int packet;
      bool propagated = false;
      try { (void)pipe.readFor(packet, 1s); }
      catch (const std::runtime_error& error) {
        propagated = std::string(error.what()) == "producer failure";
      }
      require(propagated, "producer exception was swallowed");
    }
    {
      std::promise<void> started;
      IntPipeline pipe([&]() -> std::optional<int> {
        started.set_value();
        std::this_thread::sleep_for(20ms);
        return 1;
      });
      require(started.get_future().wait_for(1s) == std::future_status::ready, "worker not started");
      pipe.stop(); // Join active computation; never leave it using destroyed dependencies.
      int packet;
      require(pipe.readFor(packet, 1s) == IntPipeline::ReadStatus::Closed,
              "stop exposed an unconsumed result");
    }
    std::cout << "single frame pipeline smoke passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
