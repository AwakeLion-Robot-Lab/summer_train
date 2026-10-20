#pragma once

#include "l1_sensor/talos/shm_layout.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace L1Sensor::talos {

struct TalosFrame {
  std::uint64_t seq{};
  std::uint64_t timestamp_ns{};
  std::uint32_t width{};
  std::uint32_t height{};
  const std::uint8_t* rgb{};
};

// Talos 是单消费者三缓冲协议。图像与四路 pose 必须由同一个实例消费，
// 否则生产者的同步屏障不会放行下一帧。
class TalosReader {
public:
  explicit TalosReader(std::string shm_dir = "/tmp");
  ~TalosReader();
  TalosReader(const TalosReader&) = delete;
  TalosReader& operator=(const TalosReader&) = delete;

  bool open();
  bool reopen();
  [[nodiscard]] bool isOpen() const noexcept { return opened_; }
  bool readFrame(TalosFrame& frame, std::chrono::milliseconds timeout);
  [[nodiscard]] const CameraInfo& cameraInfo() const noexcept { return camera_info_; }
  [[nodiscard]] const ChassisObservation& chassisObservation() const noexcept { return last_chassis_; }
  [[nodiscard]] std::optional<PoseMeta> pose(PoseIndex index) const noexcept;
  [[nodiscard]] GroundTruthBatch groundTruth() const noexcept;
  [[nodiscard]] RuntimeState runtimeState() const noexcept;
  void writeGimbalCmd(const GimbalCmd& command);
  [[nodiscard]] std::chrono::steady_clock::time_point toSteadyTime(std::uint64_t timestamp_ns) const noexcept;
  [[nodiscard]] static std::uint64_t realtimeNowNs() noexcept;

private:
  bool consumeImage(TalosFrame& frame);
  bool consumePose(PoseIndex index);
  bool tryConsumeFrame(TalosFrame& frame);
  void unmap();

  std::string shm_dir_;
  int meta_fd_{-1};
  int image_fd_{-1};
  void* meta_{};
  void* image_pool_{};
  ShmMetaRegion* region_{};
  bool opened_{};
  CameraInfo camera_info_{};
  std::array<PoseMeta, 5> last_poses_{};
  ChassisObservation last_chassis_{};
  std::chrono::steady_clock::duration steady_epoch_offset_{};
};

} // namespace L1Sensor::talos
