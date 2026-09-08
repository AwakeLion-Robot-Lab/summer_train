#include "l1_sensor/simulator/daedalus_client.hpp"
#include "l1_sensor/simulator/daedalus_protocol.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {

namespace Protocol = L1Sensor::DaedalusProtocol;

std::uint64_t nowNs()
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

void require(bool condition, const std::string& message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class Fixture {
public:
  Fixture()
  {
    const std::string suffix = std::to_string(::getpid());
    metadata_path = "/tmp/newvision_daedalus_meta_" + suffix;
    image_path = "/tmp/newvision_daedalus_image_" + suffix;
    metadata_ = createMapping(metadata_path, sizeof(Protocol::SharedMetaRegion));
    image_ = createMapping(image_path, Protocol::kImagePoolSize);

    shared = ::new (metadata_) Protocol::SharedMetaRegion{};
    pixels = static_cast<std::uint8_t*>(image_);
    shared->header.magic = Protocol::kMagic;
    shared->header.version = Protocol::kVersion;
    shared->header.created_ns = nowNs();
    shared->header.heartbeat_ns = shared->header.created_ns;
    shared->header.image_width = Protocol::kImageWidth;
    shared->header.image_height = Protocol::kImageHeight;
    shared->camera_info = {
      .timestamp_ns = shared->header.created_ns,
      .fx = 1200.0,
      .fy = 1201.0,
      .cx = 720.0,
      .cy = 540.0,
      .distortion = {0.1, -0.2, 0.0, 0.0, 0.03},
      .width = Protocol::kImageWidth,
      .height = Protocol::kImageHeight,
    };
  }

  ~Fixture()
  {
    if (shared != nullptr) {
      std::destroy_at(shared);
    }
    if (metadata_ != nullptr) {
      ::munmap(metadata_, sizeof(Protocol::SharedMetaRegion));
    }
    if (image_ != nullptr) {
      ::munmap(image_, Protocol::kImagePoolSize);
    }
    std::filesystem::remove(metadata_path);
    std::filesystem::remove(image_path);
  }

  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  void publishFrame(std::uint64_t sequence)
  {
    const std::uint64_t timestamp = nowNs();
    shared->header.heartbeat_ns = timestamp;
    shared->chassis_observation.frame_sequence = sequence;
    shared->chassis_observation.timestamp_ns = timestamp;
    shared->chassis_observation.velocity_body = {1.5F, -0.25F};
    shared->runtime_state.timestamp_ns = timestamp;
    shared->runtime_state.following = 1;

    for (std::size_t index = 0; index < Protocol::kSynchronizedPoseCount; ++index) {
      Protocol::PoseMeta pose;
      pose.frame_sequence = sequence;
      pose.timestamp_ns = timestamp;
      pose.position = {
        static_cast<float>(index),
        static_cast<float>(index + 1),
        static_cast<float>(index + 2),
      };
      pose.quaternion = {1.0F, 0.0F, 0.0F, static_cast<float>(index) * 0.1F};
      publish(shared->poses[index], pose);
    }

    // RGB [10,20,30] must be returned to OpenCV as BGR [30,20,10].
    for (std::size_t offset = 0; offset < Protocol::kImageSize; offset += 3) {
      pixels[offset] = 10;
      pixels[offset + 1] = 20;
      pixels[offset + 2] = 30;
    }

    Protocol::ImageMeta image;
    image.sequence = sequence;
    image.timestamp_ns = timestamp;
    image.width = Protocol::kImageWidth;
    image.height = Protocol::kImageHeight;
    image.buffer_id = 0;
    image.format = 0;
    publish(shared->image, image);
  }

  std::string metadata_path;
  std::string image_path;
  Protocol::SharedMetaRegion* shared = nullptr;
  std::uint8_t* pixels = nullptr;

private:
  template<typename Buffer, typename Slot>
  static void publish(Buffer& buffer, const Slot& slot)
  {
    require(buffer.write_index < buffer.slots.size(), "fixture write index is corrupt");
    buffer.slots[buffer.write_index] = slot;
    const auto old = buffer.state.exchange(
      static_cast<std::uint8_t>(buffer.write_index | Protocol::kNewDataFlag),
      std::memory_order_acq_rel);
    buffer.write_index = old & Protocol::kIndexMask;
  }

  static void* createMapping(const std::string& path, std::size_t size)
  {
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
      throw std::runtime_error("failed to create fixture file " + path);
    }
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
      ::close(fd);
      throw std::runtime_error("failed to size fixture file " + path);
    }
    void* mapping = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapping == MAP_FAILED) {
      throw std::runtime_error("failed to map fixture file " + path);
    }
    std::memset(mapping, 0, size);
    return mapping;
  }

  void* metadata_ = nullptr;
  void* image_ = nullptr;
};

}  // namespace

int main()
{
  try {
    Fixture fixture;
    L1Sensor::DaedalusClient client({fixture.metadata_path, fixture.image_path});
    require(client.connect(), "client connect failed: " + client.lastError());
    require(client.connected(), "client did not report connected state");
    require(client.isSimulatorAlive(), "fresh simulator heartbeat was rejected");
    require(client.cameraInfo().valid(), "camera intrinsics were not exposed");

    fixture.publishFrame(42);
    L1Sensor::DaedalusFrame frame;
    require(
      client.readFrame(frame, std::chrono::milliseconds{50}),
      "frame read failed: " + client.lastError());
    require(frame.sequence == 42, "image sequence was not preserved");
    require(
      frame.image_bgr.rows == static_cast<int>(Protocol::kImageHeight) &&
        frame.image_bgr.cols == static_cast<int>(Protocol::kImageWidth) &&
        frame.image_bgr.type() == CV_8UC3,
      "image dimensions/type mismatch");
    const auto pixel = frame.image_bgr.at<cv::Vec3b>(0, 0);
    require(
      pixel[0] == 30 && pixel[1] == 20 && pixel[2] == 10,
      "RGB to BGR conversion failed");
    require(
      frame.pose(L1Sensor::DaedalusPoseKind::Camera).position[0] == 3.0F,
      "camera pose was not read");
    require(
      std::abs(frame.chassis.velocity_body[0] - 1.5F) < 1e-6F,
      "chassis observation was not read");
    require(frame.auto_aim_enabled, "runtime following flag was not read");

    // The returned image owns its pixels; a later simulator write cannot mutate it.
    fixture.pixels[0] = 99;
    require(frame.image_bgr.at<cv::Vec3b>(0, 0)[2] == 10, "frame image is not owning");

    require(
      (fixture.shared->image.state.load(std::memory_order_acquire) &
       Protocol::kNewDataFlag) == 0,
      "image triple buffer was not consumed");
    for (std::size_t index = 0; index < Protocol::kSynchronizedPoseCount; ++index) {
      require(
        (fixture.shared->poses[index].state.load(std::memory_order_acquire) &
         Protocol::kNewDataFlag) == 0,
        "pose triple buffer was not consumed");
    }

    require(
      client.sendGimbalCommand(15.0F, -8.0F, 3.5F, true),
      "command send failed: " + client.lastError());
    const std::uint8_t command_state =
      fixture.shared->gimbal_command.state.load(std::memory_order_acquire);
    require(
      (command_state & Protocol::kNewDataFlag) != 0,
      "gimbal command was not published");
    const auto& command =
      fixture.shared->gimbal_command.slots[command_state & Protocol::kIndexMask];
    require(
      command.yaw_deg == 15.0F && command.pitch_deg == -8.0F &&
        command.distance_m == 3.5F && command.fire_advice == 1,
      "gimbal command fields mismatch");

    require(
      !client.readFrame(frame, std::chrono::milliseconds{1}) &&
        client.lastError().empty(),
      "no-new-frame timeout was treated as a protocol error");

    std::cout << "Daedalus client smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
