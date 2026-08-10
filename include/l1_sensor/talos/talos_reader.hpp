#pragma once

#include "l1_sensor/talos/shm_layout.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace L1Sensor::talos {

// 一帧图像的元数据与像素指针；rgb 指向共享内存图像池，
// 在后续发布覆盖同一槽位前有效，调用方应尽快拷贝。
struct TalosFrame {
  std::uint64_t seq{0};
  std::uint64_t timestamp_ns{0};
  std::uint32_t width{0};
  std::uint32_t height{0};
  const std::uint8_t* rgb{nullptr};
};

// Talos 共享内存读取器：单消费者。
class TalosReader {
public:
  explicit TalosReader(std::string shm_dir = "/tmp");
  ~TalosReader();

  TalosReader(const TalosReader&) = delete;
  TalosReader& operator=(const TalosReader&) = delete;

  // 打开并校验 magic/version；失败时返回 false。
  bool open();

  [[nodiscard]] bool isOpen() const noexcept { return opened_; }

  // 取最新帧；同时消费四路 pose 以满足仿真器同步握手
  // （否则仿真器不再发布下一帧）。
  bool readFrame(TalosFrame& frame, std::chrono::milliseconds timeout);

  [[nodiscard]] const CameraInfo& cameraInfo() const noexcept
  {
    return camera_info_;
  }
  [[nodiscard]] const ChassisObservation& chassisObservation() const noexcept
  {
    return last_chassis_;
  }
  [[nodiscard]] std::optional<PoseMeta> pose(PoseIndex index) const noexcept;
  [[nodiscard]] GroundTruthBatch groundTruth() const noexcept;

  // 生产者协议：把云台命令写回仿真器（先填 write_idx 槽再交换 state）。
  void writeGimbalCmd(const GimbalCmd& cmd);

  // SHM 的 UNIX 纳秒时间戳 → steady_clock；0 或超前过多时回退到当前时刻。
  [[nodiscard]] std::chrono::steady_clock::time_point toSteadyTime(
    std::uint64_t timestamp_ns) const noexcept;

  [[nodiscard]] static std::uint64_t realtimeNowNs() noexcept;

private:
  bool consumeImage(TalosFrame& frame);
  bool consumePose(PoseIndex index);
  bool tryConsumeFrame(TalosFrame& frame);
  void unmap();

  std::string shm_dir_;
  int meta_fd_{-1};
  int image_fd_{-1};
  void* meta_{nullptr};
  void* image_pool_{nullptr};
  ShmMetaRegion* region_{nullptr};
  bool opened_{false};
  CameraInfo camera_info_{};
  std::array<PoseMeta, 5> last_poses_{};
  ChassisObservation last_chassis_{};
  std::chrono::steady_clock::duration steady_epoch_offset_{};
};

}  // namespace L1Sensor::talos
