#pragma once

// 与 bevy_robomaster_simulator/crates/talos-ipc/src/layout.rs 的
// repr(C, align(...)) 严格对应。修改任一字段时必须同时更新两端和 ABI 测试。

#include <cstddef>
#include <cstdint>

namespace L1Sensor::talos {

inline constexpr std::uint32_t kImageWidth = 1440;
inline constexpr std::uint32_t kImageHeight = 1080;
inline constexpr std::size_t kImageSize =
  static_cast<std::size_t>(kImageWidth) * kImageHeight * 3u;
inline constexpr std::size_t kImagePoolSize = kImageSize * 3u;
inline constexpr std::uint32_t kShmMagic = 0x54414C05u;
inline constexpr std::uint32_t kShmVersion = 2u;
inline constexpr std::uint8_t kFlagNew = 0x80u;
inline constexpr std::uint8_t kIndexMask = 0x03u;
inline constexpr std::size_t kMaxGroundTruthTargets = 16;
inline constexpr std::size_t kMaxGroundTruthRunes = 4;
inline constexpr const char* kShmMetaName = "talos_ipc_meta";
inline constexpr const char* kShmImagePoolName = "talos_ipc_image_pool";

struct alignas(32) ImageMeta {
  std::uint64_t seq{};
  std::uint64_t timestamp_ns{};
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint8_t buffer_id{};
  std::uint8_t format{};
  std::uint8_t pad[6]{};
};
static_assert(sizeof(ImageMeta) == 32);

struct alignas(64) PoseMeta {
  std::uint64_t frame_seq{};
  float position[3]{};
  float quaternion[4]{1.0F, 0.0F, 0.0F, 0.0F}; // w,x,y,z
  std::uint64_t timestamp_ns{};
  std::uint8_t pad[16]{};
};
static_assert(sizeof(PoseMeta) == 64);

struct alignas(32) GimbalCmd {
  std::uint64_t timestamp_ns{};
  float yaw_deg{};
  float pitch_deg{};
  float distance_m{-1.0F};
  std::uint8_t fire_advice{};
  std::uint8_t pad[11]{};
};
static_assert(sizeof(GimbalCmd) == 32);

struct alignas(64) CameraInfo {
  std::uint64_t timestamp_ns{};
  double fx{};
  double fy{};
  double cx{};
  double cy{};
  double distortion[5]{};
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint8_t pad[24]{};
};
static_assert(sizeof(CameraInfo) == 128);

struct alignas(64) ChassisObservation {
  std::uint64_t frame_seq{};
  std::uint64_t timestamp_ns{};
  float dt_s{};
  float v_body[2]{};
  float wz_radps{};
  float wheel_linear_mps[4]{};
  float wheel_angular_radps[4]{};
  float a_body[2]{};
  float alpha_z_radps2{};
  float rpy_rad[3]{};
  float gyro_xyz_radps[3]{};
  float accel_xyz_mps2[3]{};
  std::uint8_t pad[16]{};
};
static_assert(sizeof(ChassisObservation) == 128);

struct alignas(64) ImageTripleBuffer {
  std::uint8_t state{1};
  std::uint8_t write_idx{};
  std::uint8_t read_idx{2};
  std::uint8_t pad1[61]{};
  ImageMeta slots[3]{};
};
static_assert(sizeof(ImageTripleBuffer) == 192);

struct alignas(64) PoseTripleBuffer {
  std::uint8_t state{1};
  std::uint8_t write_idx{};
  std::uint8_t read_idx{2};
  std::uint8_t pad1[61]{};
  PoseMeta slots[3]{};
};
static_assert(sizeof(PoseTripleBuffer) == 256);

struct alignas(64) GimbalTripleBuffer {
  std::uint8_t state{1};
  std::uint8_t write_idx{};
  std::uint8_t read_idx{2};
  std::uint8_t pad1[61]{};
  GimbalCmd slots[3]{};
};
static_assert(sizeof(GimbalTripleBuffer) == 192);

struct alignas(64) ShmHeader {
  std::uint32_t magic{kShmMagic};
  std::uint32_t version{kShmVersion};
  std::uint64_t created_ns{};
  std::uint64_t heartbeat_ns{};
  std::uint32_t image_width{kImageWidth};
  std::uint32_t image_height{kImageHeight};
  std::uint8_t pad[32]{};
};
static_assert(sizeof(ShmHeader) == 64);

struct alignas(32) GroundTruthTarget {
  std::uint64_t frame_seq{};
  std::uint64_t timestamp_ns{};
  std::uint8_t team{};
  std::uint8_t armor_label{};
  std::uint8_t is_outpost{};
  std::uint8_t pad1{};
  float position[3]{};
  float vyaw{};
  float yaw{};
  std::uint8_t pad[24]{};
};
static_assert(sizeof(GroundTruthTarget) == 64);

struct alignas(64) GroundTruthRune {
  std::uint64_t frame_seq{};
  std::uint64_t timestamp_ns{};
  std::uint8_t team{};
  std::uint8_t rune_mode{};
  std::uint8_t mechanism_state{};
  std::uint8_t pad1{};
  float r_center_odom[3]{};
  float radius{};
  float current_angle{};
  float v_roll{};
  std::int32_t direction{};
  float sin_amplitude{};
  float sin_omega{};
  float sin_phase{};
  float sin_offset{};
  float relative_time{};
  std::int32_t blade_id{-1};
  std::uint8_t target_activations[5]{};
  std::uint8_t pad[20]{};
};
static_assert(sizeof(GroundTruthRune) == 128);

struct alignas(64) GroundTruthBatch {
  std::uint64_t frame_seq{};
  std::uint64_t timestamp_ns{};
  std::uint32_t target_count{};
  std::uint32_t rune_count{};
  GroundTruthTarget targets[kMaxGroundTruthTargets]{};
  GroundTruthRune runes[kMaxGroundTruthRunes]{};
  std::uint8_t pad[64]{};
};
static_assert(sizeof(GroundTruthBatch) == 1664);

struct alignas(64) RuntimeState {
  std::uint64_t timestamp_ns{};
  std::uint8_t following{};
  std::uint8_t pad[55]{};
};
static_assert(sizeof(RuntimeState) == 64);

enum class PoseIndex : std::uint8_t {
  Gimbal = 0,
  Odom = 1,
  Muzzle = 2,
  Camera = 3,
  ChassisObservationLegacy = 4,
};

struct alignas(64) ShmMetaRegion {
  ShmHeader header;
  ImageTripleBuffer image;
  PoseTripleBuffer poses[5];
  GimbalTripleBuffer gimbal_cmd;
  CameraInfo camera_info;
  ChassisObservation chassis_observation;
  GroundTruthBatch ground_truth;
  RuntimeState runtime_state;
};
static_assert(offsetof(ShmMetaRegion, camera_info) == 1728);
static_assert(offsetof(ShmMetaRegion, chassis_observation) == 1856);
static_assert(offsetof(ShmMetaRegion, ground_truth) == 1984);
static_assert(offsetof(ShmMetaRegion, runtime_state) == 3648);
static_assert(sizeof(ShmMetaRegion) == 3712);

} // namespace L1Sensor::talos
