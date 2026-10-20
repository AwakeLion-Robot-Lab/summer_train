#include "l1_sensor/talos/talos_reader.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>

int main()
{
  L1Sensor::talos::TalosReader reader{"/tmp"};
  if (!reader.open()) {
    std::printf("talos_shm_smoke: simulator not running, skipped\n");
    return 0;
  }
  std::uint64_t last_seq = 0;
  int frames = 0;
  for (int attempt = 0; attempt < 30 && frames < 10; ++attempt) {
    L1Sensor::talos::TalosFrame frame;
    if (!reader.readFrame(frame, std::chrono::milliseconds{500})) continue;
    const auto pose = reader.pose(L1Sensor::talos::PoseIndex::Gimbal);
    if (!frame.rgb || !pose || pose->frame_seq != frame.seq ||
        (frames > 0 && frame.seq <= last_seq)) {
      std::printf("talos_shm_smoke: FAIL image/pose synchronization\n");
      return 1;
    }
    last_seq = frame.seq;
    ++frames;
  }
  if (frames == 0) {
    std::printf("talos_shm_smoke: FAIL no frame\n");
    return 1;
  }
  const auto info = reader.cameraInfo();
  std::printf("talos_shm_smoke: ok frames=%d seq=%llu camera=%ux%u\n",
              frames, static_cast<unsigned long long>(last_seq),
              info.width, info.height);
  return 0;
}
