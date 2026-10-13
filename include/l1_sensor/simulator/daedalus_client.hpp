#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace L1Sensor {

/**
 * @brief 仿真器每帧同步发布的几种位姿，数值即 DaedalusFrame::poses 的下标
 */
enum class DaedalusPoseKind : std::size_t {
  Gimbal = 0,
  Odom = 1,
  Muzzle = 2,
  Camera = 3,
};

/**
 * @brief 仿真器发布的一个位姿
 */
struct DaedalusPose {
  /// 对应的图像帧序号
  std::uint64_t frame_sequence = 0;

  /// 时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 位置 [x, y, z]
  std::array<float, 3> position{};

  /// 姿态四元数，w, x, y, z
  std::array<float, 4> quaternion{1.0F, 0.0F, 0.0F, 0.0F};
};

/**
 * @brief 仿真相机的内参
 */
struct DaedalusCameraInfo {
  /// 时间戳，ns；为 0 表示仿真器还没发布过
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

  /**
   * @brief 内参是否可用
   * @return 已发布、尺寸非零且焦距与主点有限、焦距为正时返回 true
   */
  [[nodiscard]] bool valid() const noexcept;
};

/**
 * @brief 仿真底盘的一帧观测，字段单位见后缀
 */
struct DaedalusChassisObservation {
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
};

/**
 * @brief 从仿真器读到的一帧完整数据
 */
struct DaedalusFrame {
  /// 仿真器发布的是 RGB8。客户端把它转成自有内存的 BGR8 cv::Mat，可以直接交给
  /// newvision/OpenCV，下一帧共享内存到来之后也仍然有效。
  cv::Mat image_bgr;

  /// 图像帧序号
  std::uint64_t sequence = 0;

  /// 仿真器侧的图像时间戳，ns
  std::uint64_t timestamp_ns = 0;

  /// 本进程收到这帧的 steady_clock 时刻
  std::chrono::steady_clock::time_point capture_time{};

  /// 与图像同步的位姿，下标见 DaedalusPoseKind
  std::array<DaedalusPose, 4> poses{};

  /// 相机内参
  DaedalusCameraInfo camera_info;

  /// 底盘观测
  DaedalusChassisObservation chassis;

  /// 仿真器侧是否处于自瞄状态
  bool auto_aim_enabled = false;

  /**
   * @brief 按种类取同步位姿
   * @param kind 位姿种类
   * @return 对应的位姿
   */
  [[nodiscard]] const DaedalusPose& pose(DaedalusPoseKind kind) const noexcept;
};

/**
 * @brief 仿真器共享内存文件的路径
 */
struct DaedalusPaths {
  /// 元数据区
  std::string metadata = "/tmp/talos_ipc_meta";

  /// 图像池
  std::string image_pool = "/tmp/talos_ipc_image_pool";
};

/**
 * @brief Daedalus 仿真器 Talos IPC v2 传输的 C++20 客户端
 *
 * 传输是单生产者单消费者：一个运行中的仿真器只能有一个 DaedalusClient 消费帧。
 */
class DaedalusClient {
public:
  /**
   * @brief 保存路径，不立即连接
   * @param paths 共享内存文件路径
   */
  explicit DaedalusClient(DaedalusPaths paths = {});

  /**
   * @brief 析构时断开连接
   */
  ~DaedalusClient();

  DaedalusClient(const DaedalusClient&) = delete;
  DaedalusClient& operator=(const DaedalusClient&) = delete;

  /**
   * @brief 移动构造，接管另一个客户端的连接
   */
  DaedalusClient(DaedalusClient&&) noexcept;

  /**
   * @brief 移动赋值，接管另一个客户端的连接
   * @return *this
   */
  DaedalusClient& operator=(DaedalusClient&&) noexcept;

  /**
   * @brief 映射共享内存并校验 magic、协议版本和图像尺寸
   *
   * 已连接时先断开再重连。
   *
   * @return 成功返回 true；失败原因见 lastError()
   */
  [[nodiscard]] bool connect() noexcept;

  /**
   * @brief 解除共享内存映射
   */
  void disconnect() noexcept;

  /**
   * @brief 是否已连接
   * @return 已连接返回 true
   */
  [[nodiscard]] bool connected() const noexcept;

  /**
   * @brief 最近一次失败的原因
   * @return 错误信息；没有错误时为空串
   */
  [[nodiscard]] std::string lastError() const;

  /**
   * @brief 仿真器是否还活着：看共享内存头里的心跳时间与当前系统时间之差
   * @param stale_after 心跳超过这么久没更新就算失联
   * @return 已连接且心跳新鲜时返回 true
   */
  [[nodiscard]] bool isSimulatorAlive(
    std::chrono::milliseconds stale_after = std::chrono::milliseconds{1000}) const noexcept;

  /**
   * @brief 读取下一帧
   *
   * 超时和协议错误都返回 false。超时不设置 lastError()；共享内存格式错误或
   * 不兼容时会设置。
   *
   * @param frame 输出的一帧数据
   * @param timeout 最长等待时间
   * @return 读到新帧返回 true
   */
  [[nodiscard]] bool readFrame(
    DaedalusFrame& frame,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{50}) noexcept;

  /**
   * @brief 向仿真器发布云台命令
   *
   * 角度沿用仿真器协议的约定：绝对角，单位度。
   *
   * @param yaw_deg 偏航角，度
   * @param pitch_deg 俯仰角，度
   * @param distance_m 目标距离，m；传 -1 表示命令无效
   * @param fire_advice 是否建议开火
   * @return 未连接、数值非法或发布失败时返回 false
   */
  [[nodiscard]] bool sendGimbalCommand(
    float yaw_deg,
    float pitch_deg,
    float distance_m,
    bool fire_advice) noexcept;

  /**
   * @brief 当前发布的相机内参
   * @return 相机内参；未连接时为默认值（valid() 为 false）
   */
  [[nodiscard]] DaedalusCameraInfo cameraInfo() const noexcept;

private:
  class Impl;

  /// 共享内存映射与协议状态
  std::unique_ptr<Impl> impl_;
};

}  // namespace L1Sensor
