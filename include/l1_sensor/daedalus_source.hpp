#pragma once

#include "l1_sensor/camera/camera_calibration.hpp"

#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace L1Sensor {

struct DaedalusSourceOptions {
  std::string meta_path{"/tmp/talos_ipc_meta"};
  std::string image_pool_path{"/tmp/talos_ipc_image_pool"};
  std::chrono::milliseconds producer_timeout{1000};
};

struct DaedalusPose {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  std::uint64_t frame_seq{0};
  std::uint64_t timestamp_ns{0};
  std::chrono::steady_clock::time_point timestamp{};
};

// 一次 read() 返回图像和四个位姿的同序号快照。bgr_image 拥有自己的内存，
// 不会在 Talos 复用共享内存槽位后失效。
struct DaedalusFrame {
  cv::Mat bgr_image;
  std::uint64_t frame_seq{0};
  std::uint64_t timestamp_ns{0};
  std::chrono::steady_clock::time_point timestamp{};

  DaedalusPose gimbal;
  DaedalusPose odom;
  DaedalusPose muzzle;
  DaedalusPose camera;
  bool following{false};
};

// Daedalus/Talos 共享内存消费者。协议是单生产者、单消费者；同一时刻只能有
// 一个 DaedalusSource 消费图像和位姿，否则三缓冲握手会被破坏。
class DaedalusSource {
public:
  [[nodiscard]] static std::unique_ptr<DaedalusSource> connect(
    const DaedalusSourceOptions& options = {},
    std::string* error = nullptr);

  ~DaedalusSource();

  DaedalusSource(const DaedalusSource&) = delete;
  DaedalusSource& operator=(const DaedalusSource&) = delete;

  // 非阻塞读取。只有图像以及 Gimbal/Odom/Muzzle/Camera 四个位姿全部到齐且
  // frame_seq 相同时才返回一帧。
  [[nodiscard]] std::optional<DaedalusFrame> read();

  [[nodiscard]] bool producerAlive() const noexcept;

  // CameraInfo 提供内参；第一帧的 camera/muzzle 相对位姿提供枪口到相机的
  // 静态平移。失败时返回 nullopt 并通过 error 给出原因。
  [[nodiscard]] std::optional<CameraCalibration> calibration(
    const DaedalusFrame& frame,
    std::string* error = nullptr) const;

  // newvision 内部统一使用弧度；此处是唯一的 radian -> degree 协议边界。
  [[nodiscard]] bool sendGimbalCommand(
    double yaw_rad,
    double pitch_rad,
    double distance_m,
    bool fire_advice) noexcept;

  // distance=-1 是 Talos 定义的无效/保持命令。
  void sendHold() noexcept;

  // 取走最近一次协议或数据错误；没有错误时返回空字符串。
  [[nodiscard]] std::string takeLastError();

private:
  struct Impl;
  explicit DaedalusSource(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace L1Sensor
