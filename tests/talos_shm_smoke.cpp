// Talos 共享内存冒烟测试：需要 Daedalus 仿真器正在运行。
// 未运行时明确跳过（返回 0）；运行时验证能连续读到帧且 seq 递增。

#include "l1_sensor/talos/talos_reader.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>

int main()
{
  L1Sensor::talos::TalosReader reader{"/tmp"};
  if (!reader.open()) {
    std::printf(
      "talos_shm_smoke: simulator not running, skipped\n");
    return 0;
  }

  const auto info = reader.cameraInfo();
  if (info.width != L1Sensor::talos::kImageWidth
      || info.height != L1Sensor::talos::kImageHeight) {
    std::printf(
      "talos_shm_smoke: FAIL camera size %ux%u\n", info.width, info.height);
    return 1;
  }

  int frames = 0;
  std::uint64_t last_seq = 0;
  for (int attempt = 0; attempt < 30 && frames < 10; ++attempt) {
    L1Sensor::talos::TalosFrame frame;
    if (!reader.readFrame(frame, std::chrono::milliseconds{500})) {
      continue;
    }
    if (frame.width != L1Sensor::talos::kImageWidth
        || frame.height != L1Sensor::talos::kImageHeight
        || frame.rgb == nullptr) {
      std::printf("talos_shm_smoke: FAIL invalid frame\n");
      return 1;
    }
    if (frames > 0 && frame.seq <= last_seq) {
      std::printf(
        "talos_shm_smoke: FAIL seq not increasing (%llu -> %llu)\n",
        static_cast<unsigned long long>(last_seq),
        static_cast<unsigned long long>(frame.seq));
      return 1;
    }
    last_seq = frame.seq;
    ++frames;
  }

  if (frames == 0) {
    std::printf("talos_shm_smoke: FAIL no frames within timeout\n");
    return 1;
  }
  std::printf(
    "talos_shm_smoke: ok, %d frames, last seq=%llu, fx=%.1f fy=%.1f "
    "cx=%.1f cy=%.1f\n",
    frames,
    static_cast<unsigned long long>(last_seq),
    info.fx, info.fy, info.cx, info.cy);
  return 0;
}
