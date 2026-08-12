#include "l1_sensor/daedalus_source.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kMagic = 0x54414C05;
constexpr std::uint32_t kVersion = 2;
constexpr std::uint32_t kWidth = 1440;
constexpr std::uint32_t kHeight = 1080;
constexpr std::size_t kImageSize =
  static_cast<std::size_t>(kWidth) * kHeight * 3;
constexpr std::size_t kPoolSize = kImageSize * 3;
constexpr std::uint8_t kFlagNew = 0x80;
constexpr std::uint8_t kIndexMask = 0x03;

struct alignas(32) ImageMeta {
  std::uint64_t seq;
  std::uint64_t timestamp_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t buffer_id;
  std::uint8_t format;
  std::uint8_t padding[6];
};

struct alignas(64) PoseMeta {
  std::uint64_t frame_seq;
  float position[3];
  float quaternion[4];
  std::uint64_t timestamp_ns;
  std::uint8_t padding[16];
};

struct alignas(32) GimbalCommand {
  std::uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;
  float distance_m;
  std::uint8_t fire_advice;
  std::uint8_t padding[11];
};

struct alignas(64) CameraInfo {
  std::uint64_t timestamp_ns;
  double fx;
  double fy;
  double cx;
  double cy;
  double distortion[5];
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t padding[24];
};

struct alignas(64) RuntimeState {
  std::uint64_t timestamp_ns;
  std::uint8_t following;
  std::uint8_t padding[55];
};

struct alignas(64) ImageBuffer {
  alignas(64) std::atomic<std::uint8_t> state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::uint8_t padding[61];
  ImageMeta slots[3];
  std::byte size_padding[32];
};

struct alignas(64) PoseBuffer {
  alignas(64) std::atomic<std::uint8_t> state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::uint8_t padding[61];
  PoseMeta slots[3];
};

struct alignas(64) GimbalBuffer {
  alignas(64) std::atomic<std::uint8_t> state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::uint8_t padding[61];
  GimbalCommand slots[3];
  std::byte size_padding[32];
};

struct alignas(64) ShmHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t created_ns;
  std::uint64_t heartbeat_ns;
  std::uint32_t image_width;
  std::uint32_t image_height;
  std::uint8_t padding[32];
};

struct MetaRegion {
  ShmHeader header;
  ImageBuffer image;
  PoseBuffer poses[5];
  GimbalBuffer command;
  CameraInfo camera_info;
  std::byte unused_channels[1792];
  RuntimeState runtime_state;
};

static_assert(sizeof(ImageMeta) == 32);
static_assert(sizeof(PoseMeta) == 64);
static_assert(sizeof(GimbalCommand) == 32);
static_assert(sizeof(ImageBuffer) == 192);
static_assert(sizeof(PoseBuffer) == 256);
static_assert(sizeof(GimbalBuffer) == 192);
static_assert(sizeof(MetaRegion) == 3712);

[[nodiscard]] std::uint64_t nowNs()
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

class TemporaryMapping {
public:
  TemporaryMapping(std::string path, std::size_t size)
    : path_(std::move(path)), size_(size)
  {
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
    require(fd_ >= 0, "cannot create " + path_);
    require(::ftruncate(fd_, static_cast<off_t>(size_)) == 0,
            "cannot size " + path_);
    data_ = ::mmap(
      nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    require(data_ != MAP_FAILED, "cannot mmap " + path_);
    std::memset(data_, 0, size_);
  }

  ~TemporaryMapping()
  {
    if (data_ != nullptr && data_ != MAP_FAILED) {
      ::munmap(data_, size_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
    std::filesystem::remove(path_);
  }

  [[nodiscard]] void* data() const { return data_; }

private:
  std::string path_;
  std::size_t size_{0};
  int fd_{-1};
  void* data_{nullptr};
};

template<typename Buffer>
void initialize(Buffer& buffer)
{
  new (&buffer.state) std::atomic<std::uint8_t>{1};
  buffer.write_index = 0;
  buffer.read_index = 2;
}

template<typename Buffer, typename Slot>
void publish(Buffer& buffer, const Slot& value)
{
  buffer.slots[buffer.write_index] = value;
  const std::uint8_t old = buffer.state.exchange(
    static_cast<std::uint8_t>(buffer.write_index | kFlagNew),
    std::memory_order_acq_rel);
  buffer.write_index = old & kIndexMask;
}

template<typename Buffer, typename Slot>
Slot consume(Buffer& buffer)
{
  const std::uint8_t state = buffer.state.load(std::memory_order_acquire);
  require((state & kFlagNew) != 0, "expected a published command");
  const std::uint8_t ready = state & kIndexMask;
  buffer.state.store(buffer.read_index, std::memory_order_release);
  buffer.read_index = ready;
  return buffer.slots[ready];
}

}  // namespace

int main()
{
  try {
    const std::string stem =
      "/tmp/newvision_daedalus_source_" + std::to_string(::getpid());
    const std::string meta_path = stem + "_meta";
    const std::string pool_path = stem + "_pool";
    TemporaryMapping meta_mapping(meta_path, sizeof(MetaRegion));
    TemporaryMapping pool_mapping(pool_path, kPoolSize);

    auto* meta = static_cast<MetaRegion*>(meta_mapping.data());
    meta->header.magic = kMagic;
    meta->header.version = kVersion;
    meta->header.heartbeat_ns = nowNs();
    meta->header.image_width = kWidth;
    meta->header.image_height = kHeight;
    initialize(meta->image);
    for (auto& pose : meta->poses) {
      initialize(pose);
    }
    initialize(meta->command);

    meta->camera_info.timestamp_ns = nowNs();
    meta->camera_info.fx = 1303.675323681;
    meta->camera_info.fy = 1303.675323681;
    meta->camera_info.cx = 720.0;
    meta->camera_info.cy = 540.0;
    meta->camera_info.width = kWidth;
    meta->camera_info.height = kHeight;
    meta->runtime_state.timestamp_ns = nowNs();
    meta->runtime_state.following = 1;

    auto* image_pool = static_cast<std::uint8_t*>(pool_mapping.data());
    image_pool[0] = 255;
    image_pool[1] = 0;
    image_pool[2] = 0;

    constexpr std::uint64_t kFrameSequence = 42;
    const std::uint64_t timestamp_ns = nowNs();
    ImageMeta image{};
    image.seq = kFrameSequence;
    image.timestamp_ns = timestamp_ns;
    image.width = kWidth;
    image.height = kHeight;
    image.buffer_id = 0;
    image.format = 0;

    PoseMeta pose{};
    pose.frame_seq = kFrameSequence;
    pose.quaternion[0] = 1.0F;
    pose.timestamp_ns = timestamp_ns;
    publish(meta->poses[0], pose);
    publish(meta->poses[1], pose);
    pose.position[0] = 0.010704F;
    pose.position[1] = 0.001831F;
    pose.position[2] = 0.110288F;
    publish(meta->poses[2], pose);
    pose.position[0] = 0.060025163F;
    pose.position[1] = 0.002016812F;
    pose.position[2] = 0.196342126F;
    publish(meta->poses[3], pose);
    publish(meta->image, image);

    L1Sensor::DaedalusSourceOptions options;
    options.meta_path = meta_path;
    options.image_pool_path = pool_path;
    std::string error;
    auto source = L1Sensor::DaedalusSource::connect(options, &error);
    require(source != nullptr, "connect failed: " + error);
    require(source->producerAlive(), "fresh heartbeat was rejected");

    const auto frame = source->read();
    require(frame.has_value(), "synchronized frame was not returned");
    require(frame->frame_seq == kFrameSequence, "frame sequence changed");
    require(frame->following, "runtime F5 state was not propagated");
    require(frame->bgr_image.rows == static_cast<int>(kHeight) &&
            frame->bgr_image.cols == static_cast<int>(kWidth),
            "image dimensions changed");
    const cv::Vec3b first_pixel = frame->bgr_image.at<cv::Vec3b>(0, 0);
    require(first_pixel == cv::Vec3b(0, 0, 255),
            "RGB image was not converted to BGR");

    const auto calibration = source->calibration(*frame, &error);
    require(calibration.has_value(), "calibration failed: " + error);
    require(calibration->barrelExtrinsicsReady(),
            "barrel extrinsics were not generated");
    const Eigen::Vector3d expected_translation{
      0.049321163, 0.000185812, 0.086054126};
    require(
      (calibration->T_barrel_camera->translation() - expected_translation)
          .norm() < 1e-6,
      "camera-minus-muzzle translation is wrong");

    require(source->sendGimbalCommand(
              std::numbers::pi / 2.0,
              -std::numbers::pi / 6.0,
              3.0, true),
            "valid command was rejected");
    const GimbalCommand command =
      consume<GimbalBuffer, GimbalCommand>(meta->command);
    require(std::abs(command.yaw_deg - 90.0F) < 1e-4F,
            "yaw radian-to-degree conversion failed");
    require(std::abs(command.pitch_deg + 30.0F) < 1e-4F,
            "pitch sign or radian-to-degree conversion failed");
    require(command.distance_m == 3.0F && command.fire_advice == 1,
            "command payload changed");

    source->sendHold();
    const GimbalCommand hold =
      consume<GimbalBuffer, GimbalCommand>(meta->command);
    require(hold.distance_m == -1.0F && hold.fire_advice == 0,
            "hold command is not safe");

    std::cout << "daedalus_source_smoke passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "daedalus_source_smoke failed: "
              << exception.what() << '\n';
    return 1;
  }
}
