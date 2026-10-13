#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

/**
 * @file
 * @brief Daedalus 仿真器 Talos IPC v2 的共享内存布局
 *
 * 这里的常量和内存布局逐字节对应 daedalus/crates/talos-ipc v2。仿真器用的是
 * /tmp 下的普通文件加 mmap(MAP_SHARED)，不是 POSIX shm_open 的命名共享内存。
 * 每个结构体后面的 static_assert 守着 ABI，改动任何字段都会在编译期报错。
 */
namespace L1Sensor::DaedalusProtocol {

/// 元数据区头部的 magic
inline constexpr std::uint32_t kMagic = 0x54414C05;

/// 协议版本
inline constexpr std::uint32_t kVersion = 2;

/// 图像宽，pixel
inline constexpr std::uint32_t kImageWidth = 1440;

/// 图像高，pixel
inline constexpr std::uint32_t kImageHeight = 1080;

/// 图像通道数
inline constexpr std::uint32_t kImageChannels = 3;

/// 单张图像的字节数
inline constexpr std::size_t kImageSize =
  static_cast<std::size_t>(kImageWidth) * kImageHeight * kImageChannels;

/// 图像池总字节数（三缓冲）
inline constexpr std::size_t kImagePoolSize = kImageSize * 3;

/// 三缓冲 state 字节里的"有新数据"标志位
inline constexpr std::uint8_t kNewDataFlag = 0x80;

/// 三缓冲 state 字节里的槽位下标掩码
inline constexpr std::uint8_t kIndexMask = 0x03;

/// 共享内存里位姿三缓冲的个数
inline constexpr std::size_t kPoseCount = 5;

/// 其中与图像同步发布的位姿个数，见 DaedalusPoseKind
inline constexpr std::size_t kSynchronizedPoseCount = 4;

/**
 * @brief 一张图像的元数据
 */
struct alignas(32) ImageMeta {
  /// 图像帧序号
  std::uint64_t sequence = 0;

  /// 时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 图像宽，pixel
  std::uint32_t width = 0;

  /// 图像高，pixel
  std::uint32_t height = 0;

  /// 图像在图像池中的槽位
  std::uint8_t buffer_id = 0;

  /// 像素格式：0=RGB8, 1=BGR8, 2=GRAY8
  std::uint8_t format = 0;

  /// 对齐填充
  std::array<std::uint8_t, 6> padding{};
};
static_assert(sizeof(ImageMeta) == 32);

/**
 * @brief 一个位姿
 */
struct alignas(64) PoseMeta {
  /// 对应的图像帧序号
  std::uint64_t frame_sequence = 0;

  /// 位置 [x, y, z]
  std::array<float, 3> position{};

  /// 姿态四元数，w, x, y, z
  std::array<float, 4> quaternion{};

  /// 时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 对齐填充
  std::array<std::uint8_t, 16> padding{};
};
static_assert(sizeof(PoseMeta) == 64);

/**
 * @brief 视觉发给仿真器的云台命令
 */
struct alignas(32) GimbalCommand {
  /// 时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 偏航角，绝对角，度
  float yaw_deg = 0.0F;

  /// 俯仰角，绝对角，度
  float pitch_deg = 0.0F;

  /// 目标距离，m；-1 表示命令无效
  float distance_m = -1.0F;

  /// 是否建议开火
  std::uint8_t fire_advice = 0;

  /// 对齐填充
  std::array<std::uint8_t, 11> padding{};
};
static_assert(sizeof(GimbalCommand) == 32);

/**
 * @brief 仿真相机内参
 */
struct alignas(64) CameraInfo {
  /// 时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 焦距 x，pixel
  double fx = 0.0;

  /// 焦距 y，pixel
  double fy = 0.0;

  /// 主点 x，pixel
  double cx = 0.0;

  /// 主点 y，pixel
  double cy = 0.0;

  /// 畸变系数 [k1, k2, p1, p2, k3]
  std::array<double, 5> distortion{};

  /// 图像宽，pixel
  std::uint32_t width = 0;

  /// 图像高，pixel
  std::uint32_t height = 0;

  /// 对齐填充
  std::array<std::uint8_t, 24> padding{};
};
static_assert(sizeof(CameraInfo) == 128);

/**
 * @brief 仿真底盘观测，字段含义同 DaedalusChassisObservation
 */
struct alignas(64) ChassisObservation {
  /// 对应的图像帧序号
  std::uint64_t frame_sequence = 0;

  /// 时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 与上一帧的时间间隔，s
  float dt_s = 0.0F;

  /// 车体系线速度 [x, y]
  std::array<float, 2> velocity_body{};

  /// 偏航角速度，rad/s
  float yaw_rate_rad_s = 0.0F;

  /// 四个轮子的线速度，m/s
  std::array<float, 4> wheel_linear_m_s{};

  /// 四个轮子的角速度，rad/s
  std::array<float, 4> wheel_angular_rad_s{};

  /// 车体系线加速度 [x, y]
  std::array<float, 2> acceleration_body{};

  /// 偏航角加速度，rad/s²
  float yaw_acceleration_rad_s2 = 0.0F;

  /// 姿态 [roll, pitch, yaw]，rad
  std::array<float, 3> rpy_rad{};

  /// 陀螺仪角速度，rad/s
  std::array<float, 3> gyro_rad_s{};

  /// 加速度计读数，m/s²
  std::array<float, 3> acceleration_m_s2{};

  /// 对齐填充
  std::array<std::uint8_t, 16> padding{};
};
static_assert(sizeof(ChassisObservation) == 128);

/**
 * @brief 仿真器运行状态
 */
struct alignas(64) RuntimeState {
  /// 时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 仿真器侧是否处于自瞄跟随
  std::uint8_t following = 0;

  /// 对齐填充
  std::array<std::uint8_t, 55> padding{};
};
static_assert(sizeof(RuntimeState) == 64);

/**
 * @brief 图像元数据的三缓冲
 */
struct alignas(64) ImageTripleBuffer {
  /// 低位为中间槽下标，kNewDataFlag 表示有新数据
  std::atomic<std::uint8_t> state{1};

  /// 生产者正在写的槽
  std::uint8_t write_index = 0;

  /// 消费者正在读的槽
  std::uint8_t read_index = 2;

  /// 对齐填充
  std::array<std::uint8_t, 61> padding{};

  /// 三个槽
  std::array<ImageMeta, 3> slots{};
};
static_assert(sizeof(ImageTripleBuffer) == 192);

/**
 * @brief 位姿的三缓冲，字段含义同 ImageTripleBuffer
 */
struct alignas(64) PoseTripleBuffer {
  /// 低位为中间槽下标，kNewDataFlag 表示有新数据
  std::atomic<std::uint8_t> state{1};

  /// 生产者正在写的槽
  std::uint8_t write_index = 0;

  /// 消费者正在读的槽
  std::uint8_t read_index = 2;

  /// 对齐填充
  std::array<std::uint8_t, 61> padding{};

  /// 三个槽
  std::array<PoseMeta, 3> slots{};
};
static_assert(sizeof(PoseTripleBuffer) == 256);

/**
 * @brief 云台命令的三缓冲，字段含义同 ImageTripleBuffer；这里视觉是生产者
 */
struct alignas(64) GimbalTripleBuffer {
  /// 低位为中间槽下标，kNewDataFlag 表示有新数据
  std::atomic<std::uint8_t> state{1};

  /// 生产者正在写的槽
  std::uint8_t write_index = 0;

  /// 消费者正在读的槽
  std::uint8_t read_index = 2;

  /// 对齐填充
  std::array<std::uint8_t, 61> padding{};

  /// 三个槽
  std::array<GimbalCommand, 3> slots{};
};
static_assert(sizeof(GimbalTripleBuffer) == 192);

/**
 * @brief 元数据区头部
 */
struct alignas(64) SharedHeader {
  /// 必须等于 kMagic
  std::uint32_t magic = 0;

  /// 必须等于 kVersion
  std::uint32_t version = 0;

  /// 共享内存创建时刻，系统时间 ns
  std::uint64_t created_ns = 0;

  /// 仿真器心跳，系统时间 ns；为 0 表示从未写过
  std::uint64_t heartbeat_ns = 0;

  /// 图像宽，pixel
  std::uint32_t image_width = 0;

  /// 图像高，pixel
  std::uint32_t image_height = 0;

  /// 对齐填充
  std::array<std::uint8_t, 32> padding{};
};
static_assert(sizeof(SharedHeader) == 64);

/// 实时自瞄连接器暂时用不到真值。这里仍把这段保留区按原长度留着，后面所有
/// 字段的偏移才与仿真器 ABI 一致。
inline constexpr std::size_t kGroundTruthRegionSize = 1664;

/**
 * @brief 整个元数据区的内存布局，映射 /tmp/talos_ipc_meta
 */
struct alignas(64) SharedMetaRegion {
  /// 头部
  SharedHeader header;

  /// 图像元数据
  ImageTripleBuffer image;

  /// 位姿，前 kSynchronizedPoseCount 个与图像同步
  std::array<PoseTripleBuffer, kPoseCount> poses;

  /// 云台命令
  GimbalTripleBuffer gimbal_command;

  /// 相机内参
  CameraInfo camera_info;

  /// 底盘观测
  ChassisObservation chassis_observation;

  /// 真值保留区，未解析
  std::array<std::byte, kGroundTruthRegionSize> ground_truth;

  /// 运行状态
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
