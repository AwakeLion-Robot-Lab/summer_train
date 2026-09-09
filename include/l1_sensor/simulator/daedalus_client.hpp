#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace L1Sensor {

enum class DaedalusPoseKind : std::size_t {
  Gimbal = 0,
  Odom = 1,
  Muzzle = 2,
  Camera = 3,
};

struct DaedalusPose {
  std::uint64_t frame_sequence = 0;
  std::uint64_t timestamp_ns = 0;
  std::array<float, 3> position{};
  std::array<float, 4> quaternion{1.0F, 0.0F, 0.0F, 0.0F};  // w, x, y, z
};

struct DaedalusCameraInfo {
  std::uint64_t timestamp_ns = 0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  std::array<double, 5> distortion{};
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  [[nodiscard]] bool valid() const noexcept;
};

struct DaedalusChassisObservation {
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
};

struct DaedalusFrame {
  // The simulator publishes RGB8.  The client converts it to an owning BGR8
  // cv::Mat so it can be passed directly to newvision/OpenCV and remains valid
  // after the next shared-memory frame arrives.
  cv::Mat image_bgr;
  std::uint64_t sequence = 0;
  std::uint64_t timestamp_ns = 0;
  std::chrono::steady_clock::time_point capture_time{};
  std::array<DaedalusPose, 4> poses{};
  DaedalusCameraInfo camera_info;
  DaedalusChassisObservation chassis;
  bool auto_aim_enabled = false;

  [[nodiscard]] const DaedalusPose& pose(DaedalusPoseKind kind) const noexcept;
};

struct DaedalusPaths {
  std::string metadata = "/tmp/talos_ipc_meta";
  std::string image_pool = "/tmp/talos_ipc_image_pool";
};

// C++20 client for Daedalus' Talos IPC v2 transport.  The transport is SPSC:
// exactly one DaedalusClient may consume frames from a running simulator.
class DaedalusClient {
public:
  explicit DaedalusClient(DaedalusPaths paths = {});
  ~DaedalusClient();

  DaedalusClient(const DaedalusClient&) = delete;
  DaedalusClient& operator=(const DaedalusClient&) = delete;
  DaedalusClient(DaedalusClient&&) noexcept;
  DaedalusClient& operator=(DaedalusClient&&) noexcept;

  [[nodiscard]] bool connect() noexcept;
  void disconnect() noexcept;
  [[nodiscard]] bool connected() const noexcept;
  [[nodiscard]] std::string lastError() const;

  [[nodiscard]] bool isSimulatorAlive(
    std::chrono::milliseconds stale_after = std::chrono::milliseconds{1000}) const noexcept;

  // Returns false on timeout as well as on a protocol error.  A timeout leaves
  // lastError() empty; malformed/incompatible shared memory sets it.
  [[nodiscard]] bool readFrame(
    DaedalusFrame& frame,
    std::chrono::milliseconds timeout = std::chrono::milliseconds{50}) noexcept;

  // Angles use the simulator protocol convention: absolute degrees.  Set
  // distance_m to -1 to mark the command invalid.
  [[nodiscard]] bool sendGimbalCommand(
    float yaw_deg,
    float pitch_deg,
    float distance_m,
    bool fire_advice) noexcept;

  [[nodiscard]] DaedalusCameraInfo cameraInfo() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace L1Sensor
