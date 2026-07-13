#include "latesbuffer.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

int main()
{
  tools::LatestBuffer<int> buffer;
  buffer.write(1);
  buffer.write(2);

  int value = 0;
  if (!buffer.read(value) || value != 2) {
    std::cerr << "LatestBuffer did not return the newest value\n";
    return 1;
  }

  if (buffer.droppedCount() != 1) {
    std::cerr << "LatestBuffer dropped count mismatch\n";
    return 2;
  }

  buffer.close();
  if (buffer.read(value)) {
    std::cerr << "LatestBuffer should be empty after close\n";
    return 3;
  }

  tools::LatestBuffer<int> lifecycle_buffer;
  if (lifecycle_buffer.readFor(value, std::chrono::milliseconds{1}) !=
      tools::LatestBuffer<int>::ReadStatus::Timeout) {
    std::cerr << "LatestBuffer should time out when no value is available\n";
    return 4;
  }

  std::atomic<int> read_status{
    static_cast<int>(tools::LatestBuffer<int>::ReadStatus::Timeout)};
  std::thread reader([&] {
    read_status.store(static_cast<int>(
      lifecycle_buffer.readFor(value, std::chrono::seconds{1})));
  });
  lifecycle_buffer.close();
  reader.join();

  if (read_status.load() !=
      static_cast<int>(tools::LatestBuffer<int>::ReadStatus::Closed)) {
    std::cerr << "LatestBuffer close did not wake a waiting reader\n";
    return 5;
  }

  std::cout << "LatestBuffer smoke test passed\n";
  return 0;
}
