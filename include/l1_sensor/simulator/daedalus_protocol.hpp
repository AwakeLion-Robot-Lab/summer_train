#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace L1Sensor::DaedalusProtocol {

// These constants and layouts mirror daedalus/crates/talos-ipc v2.  The
// simulator uses ordinary files under /tmp and mmap(MAP_SHARED), not POSIX
// shm_open names.
inline constexpr std::uint32_t kMagic = 0x54414C05;
inline constexpr std::uint32_t kVersion = 2;
inline constexpr std::uint32_t kImageWidth = 1440;
inline constexpr std::uint32_t kImageHeight = 1080;
inline constexpr std::uint32_t kImageChannels = 3;
inline constexpr std::size_t kImageSize =
  static_cast<std::size_t>(kImageWidth) * kImageHeight * kImageChannels;
inline constexpr std::size_t kImagePoolSize = kImageSize * 3;

inline constexpr std::uint8_t kNewDataFlag = 0x80;
inline constexpr std::uint8_t kIndexMask = 0x03;
inline constexpr std::size_t kPoseCount = 5;
inline constexpr std::size_t kSynchronizedPoseCount = 4;

struct alignas(32) ImageMeta {
  std::uint64_t sequence = 0;
  std::uint64_t timestamp_ns = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint8_t buffer_id = 0;
  std::uint8_t format = 0;  // 0=RGB8, 1=BGR8, 2=GRAY8
  std::array<std::uint8_t, 6> padding{};
};
static_assert(sizeof(ImageMeta) == 32);

struct alignas(64) PoseMeta {
  std::uint64_t frame_sequence = 0;
  std::array<float, 3> position{};
  std::array<float, 4> quaternion{};  // w, x, y, z
  std::uint64_t timestamp_ns = 0;
  std::array<std::uint8_t, 16> padding{};
};
static_assert(sizeof(PoseMeta) == 64);

struct alignas(32) GimbalCommand {
  std::uint64_t timestamp_ns = 0;
  float yaw_deg = 0.0F;
  float pitch_deg = 0.0F;
  float distance_m = -1.0F;
  std::uint8_t fire_advice = 0;
  std::array<std::uint8_t, 11> padding{};
};
static_assert(sizeof(GimbalCommand) == 32);

struct alignas(64) CameraInfo {
  std::uint64_t timestamp_ns = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::array<double, 5> distortion{};
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::array<std::uint8_t, 24> padding{};
};
static_assert(sizeof(CameraInfo) == 128);

struct alignas(64) ChassisObservation {
  std::uint64_t frame_sequence = 0;
  std::uint64_t timestamp_ns = 0;
  float dt_s = 0.0F;
  std::array<float, 2> velocity_body{};
  float yaw_rate_rad_s = 0.0F;
  std::array<float, 4> wheel_linear_m_s{};
  std::array<float, 4> wheel_angular_rad_s{};
  std::array<float, 2> acceleration_body{};
  float yaw_acceleration_rad_s2 = 0.0F;
  std::array<float, 3> rpy_rad{};
  std::array<float, 3> gyro_rad_s{};
  std::array<float, 3> acceleration_m_s2{};
  std::array<std::uint8_t, 16> padding{};
};
static_assert(sizeof(ChassisObservation) == 128);

struct alignas(64) RuntimeState {
  std::uint64_t timestamp_ns = 0;
  std::uint8_t following = 0;
  std::array<std::uint8_t, 55> padding{};
};
static_assert(sizeof(RuntimeState) == 64);

struct alignas(64) ImageTripleBuffer {
  std::atomic<std::uint8_t> state{1};
  std::uint8_t write_index = 0;
  std::uint8_t read_index = 2;
  std::array<std::uint8_t, 61> padding{};
  std::array<ImageMeta, 3> slots{};
};
static_assert(sizeof(ImageTripleBuffer) == 192);

struct alignas(64) PoseTripleBuffer {
  std::atomic<std::uint8_t> state{1};
  std::uint8_t write_index = 0;
  std::uint8_t read_index = 2;
  std::array<std::uint8_t, 61> padding{};
  std::array<PoseMeta, 3> slots{};
};
static_assert(sizeof(PoseTripleBuffer) == 256);

struct alignas(64) GimbalTripleBuffer {
  std::atomic<std::uint8_t> state{1};
  std::uint8_t write_index = 0;
  std::uint8_t read_index = 2;
  std::array<std::uint8_t, 61> padding{};
  std::array<GimbalCommand, 3> slots{};
};
static_assert(sizeof(GimbalTripleBuffer) == 192);

struct alignas(64) SharedHeader {
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint64_t created_ns = 0;
  std::uint64_t heartbeat_ns = 0;
  std::uint32_t image_width = 0;
  std::uint32_t image_height = 0;
  std::array<std::uint8_t, 32> padding{};
};
static_assert(sizeof(SharedHeader) == 64);

// Ground truth is not needed by the live auto-aim connector yet.  Keeping the
// exact reserved span here still makes every following offset ABI-compatible.
inline constexpr std::size_t kGroundTruthRegionSize = 1664;

struct alignas(64) SharedMetaRegion {
  SharedHeader header;
  ImageTripleBuffer image;
  std::array<PoseTripleBuffer, kPoseCount> poses;
  GimbalTripleBuffer gimbal_command;
  CameraInfo camera_info;
  ChassisObservation chassis_observation;
  std::array<std::byte, kGroundTruthRegionSize> ground_truth;
  RuntimeState runtime_state;
};

static_assert(sizeof(std::atomic<std::uint8_t>) == sizeof(std::uint8_t));
static_assert(std::atomic<std::uint8_t>::is_always_lock_free);
static_assert(std::is_standard_layout_v<SharedMetaRegion>);
static_assert(offsetof(SharedMetaRegion, image) == 64);
static_assert(offsetof(SharedMetaRegion, poses) == 256);
static_assert(offsetof(SharedMetaRegion, gimbal_command) == 1536);
static_assert(offsetof(SharedMetaRegion, camera_info) == 1728);
static_assert(offsetof(SharedMetaRegion, chassis_observation) == 1856);
static_assert(offsetof(SharedMetaRegion, ground_truth) == 1984);
static_assert(offsetof(SharedMetaRegion, runtime_state) == 3648);
static_assert(sizeof(SharedMetaRegion) == 3712);

}  // namespace L1Sensor::DaedalusProtocol
