#include "latesbuffer.hpp"

#include <iostream>

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

  std::cout << "LatestBuffer smoke test passed\n";
  return 0;
}
