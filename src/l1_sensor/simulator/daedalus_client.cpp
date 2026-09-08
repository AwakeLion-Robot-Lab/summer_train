#include "l1_sensor/simulator/daedalus_client.hpp"

#include "l1_sensor/simulator/daedalus_protocol.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace L1Sensor {
namespace {

namespace Protocol = DaedalusProtocol;

std::uint64_t systemNowNs() noexcept
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

std::chrono::steady_clock::time_point toSteadyTime(std::uint64_t timestamp_ns) noexcept
{
  const auto steady_now = std::chrono::steady_clock::now();
  const std::uint64_t system_now = systemNowNs();
  constexpr auto kMaxDelta = std::chrono::hours{24};
  const auto max_delta_ns = static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(kMaxDelta).count());

  // Daedalus timestamps use UNIX epoch while the newvision pipeline uses a
  // monotonic clock.  Sampling both clocks together preserves frame age
  // without making the rest of the pipeline sensitive to wall-clock jumps.
  if (timestamp_ns <= system_now) {
    const std::uint64_t age_ns = system_now - timestamp_ns;
    if (age_ns > max_delta_ns) {
      return steady_now;
    }
    return steady_now - std::chrono::nanoseconds{age_ns};
  }

  const std::uint64_t ahead_ns = timestamp_ns - system_now;
  if (ahead_ns > max_delta_ns) {
    return steady_now;
  }
  return steady_now + std::chrono::nanoseconds{ahead_ns};
}

class MappedRegion {
public:
  MappedRegion() = default;
  ~MappedRegion() { reset(); }

  MappedRegion(const MappedRegion&) = delete;
  MappedRegion& operator=(const MappedRegion&) = delete;

  MappedRegion(MappedRegion&& other) noexcept
  {
    *this = std::move(other);
  }

  MappedRegion& operator=(MappedRegion&& other) noexcept
  {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  bool map(const std::string& path, std::size_t minimum_size, std::string& error) noexcept
  {
    reset();

    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
      error = "cannot open " + path + ": " + std::strerror(errno);
      return false;
    }

    struct stat status {};
    if (::fstat(fd, &status) != 0) {
      error = "cannot stat " + path + ": " + std::strerror(errno);
      ::close(fd);
      return false;
    }
    if (status.st_size < 0 || static_cast<std::uint64_t>(status.st_size) < minimum_size) {
      std::ostringstream message;
      message << path << " is too small: expected at least " << minimum_size
              << " bytes, got " << status.st_size;
      error = message.str();
      ::close(fd);
      return false;
    }

    void* mapping = ::mmap(
      nullptr, minimum_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int map_errno = errno;
    ::close(fd);
    if (mapping == MAP_FAILED) {
      error = "cannot mmap " + path + ": " + std::strerror(map_errno);
      return false;
    }

    data_ = mapping;
    size_ = minimum_size;
    return true;
  }

  void reset() noexcept
  {
    if (data_ != nullptr) {
      ::munmap(data_, size_);
      data_ = nullptr;
      size_ = 0;
    }
  }

  template<typename T>
  T* as() noexcept
  {
    return static_cast<T*>(data_);
  }

  std::uint8_t* bytes() noexcept
  {
    return static_cast<std::uint8_t*>(data_);
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

template<typename Buffer>
using BufferSlot = typename decltype(Buffer::slots)::value_type;

template<typename Buffer>
std::optional<BufferSlot<Buffer>> consumeLatest(
  Buffer& buffer,
  std::string_view channel,
  std::string& error) noexcept
{
  for (int attempt = 0; attempt < 2; ++attempt) {
    std::uint8_t expected = buffer.state.load(std::memory_order_acquire);
    if ((expected & Protocol::kNewDataFlag) == 0) {
      return std::nullopt;
    }

    const std::uint8_t ready_index = expected & Protocol::kIndexMask;
    if (ready_index >= buffer.slots.size() || buffer.read_index >= buffer.slots.size()) {
      error = std::string{channel} + " triple-buffer index is corrupt";
      return std::nullopt;
    }

    const std::uint8_t desired = buffer.read_index;
    if (buffer.state.compare_exchange_strong(
          expected,
          desired,
          std::memory_order_acq_rel,
          std::memory_order_acquire)) {
      buffer.read_index = ready_index;
      return buffer.slots[ready_index];
    }
  }

  return std::nullopt;
}

template<typename Buffer, typename Slot>
bool publishLatest(
  Buffer& buffer,
  const Slot& value,
  std::string_view channel,
  std::string& error) noexcept
{
  const std::uint8_t state = buffer.state.load(std::memory_order_acquire);
  if ((state & Protocol::kIndexMask) >= buffer.slots.size() ||
      buffer.write_index >= buffer.slots.size()) {
    error = std::string{channel} + " triple-buffer index is corrupt";
    return false;
  }

  buffer.slots[buffer.write_index] = value;
  const std::uint8_t old = buffer.state.exchange(
    static_cast<std::uint8_t>(buffer.write_index | Protocol::kNewDataFlag),
    std::memory_order_acq_rel);
  buffer.write_index = old & Protocol::kIndexMask;
  return true;
}

DaedalusCameraInfo convertCameraInfo(const Protocol::CameraInfo& source) noexcept
{
  return {
    .timestamp_ns = source.timestamp_ns,
    .fx = source.fx,
    .fy = source.fy,
    .cx = source.cx,
    .cy = source.cy,
    .distortion = source.distortion,
    .width = source.width,
    .height = source.height,
  };
}

DaedalusChassisObservation convertChassis(
  const Protocol::ChassisObservation& source) noexcept
{
  return {
    .frame_sequence = source.frame_sequence,
    .timestamp_ns = source.timestamp_ns,
    .dt_s = source.dt_s,
    .velocity_body = source.velocity_body,
    .yaw_rate_rad_s = source.yaw_rate_rad_s,
    .wheel_linear_m_s = source.wheel_linear_m_s,
    .wheel_angular_rad_s = source.wheel_angular_rad_s,
    .acceleration_body = source.acceleration_body,
    .yaw_acceleration_rad_s2 = source.yaw_acceleration_rad_s2,
    .rpy_rad = source.rpy_rad,
    .gyro_rad_s = source.gyro_rad_s,
    .acceleration_m_s2 = source.acceleration_m_s2,
  };
}

}  // namespace

class DaedalusClient::Impl {
public:
  explicit Impl(DaedalusPaths paths) : paths_(std::move(paths)) {}

  bool connect() noexcept
  {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    disconnectUnlocked();
    clearError();

    std::string error;
    if (!metadata_region_.map(
          paths_.metadata, sizeof(Protocol::SharedMetaRegion), error)) {
      setError(std::move(error));
      return false;
    }
    if (!image_region_.map(paths_.image_pool, Protocol::kImagePoolSize, error)) {
      metadata_region_.reset();
      setError(std::move(error));
      return false;
    }

    Protocol::SharedMetaRegion* metadata =
      metadata_region_.as<Protocol::SharedMetaRegion>();
    if (metadata->header.magic != Protocol::kMagic) {
      std::ostringstream message;
      message << "Daedalus metadata magic mismatch: expected 0x" << std::hex
              << Protocol::kMagic << ", got 0x" << metadata->header.magic;
      disconnectUnlocked();
      setError(message.str());
      return false;
    }
    if (metadata->header.version != Protocol::kVersion) {
      std::ostringstream message;
      message << "Daedalus protocol version mismatch: client supports "
              << Protocol::kVersion << ", simulator publishes "
              << metadata->header.version;
      disconnectUnlocked();
      setError(message.str());
      return false;
    }
    if (metadata->header.image_width != Protocol::kImageWidth ||
        metadata->header.image_height != Protocol::kImageHeight) {
      std::ostringstream message;
      message << "Daedalus image layout mismatch: expected "
              << Protocol::kImageWidth << 'x' << Protocol::kImageHeight << ", got "
              << metadata->header.image_width << 'x'
              << metadata->header.image_height;
      disconnectUnlocked();
      setError(message.str());
      return false;
    }

    metadata_ = metadata;
    image_pool_ = image_region_.bytes();
    return true;
  }

  void disconnect() noexcept
  {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    disconnectUnlocked();
  }

  bool connected() const noexcept
  {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    return metadata_ != nullptr && image_pool_ != nullptr;
  }

  bool isSimulatorAlive(std::chrono::milliseconds stale_after) const noexcept
  {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    if (metadata_ == nullptr || stale_after.count() < 0) {
      return false;
    }

    const std::uint64_t heartbeat = metadata_->header.heartbeat_ns;
    if (heartbeat == 0) {
      return false;
    }

    const auto threshold_count =
      std::chrono::duration_cast<std::chrono::nanoseconds>(stale_after).count();
    if (threshold_count < 0) {
      return false;
    }
    const std::uint64_t threshold = static_cast<std::uint64_t>(threshold_count);
    const std::uint64_t now = systemNowNs();
    const std::uint64_t delta = now >= heartbeat ? now - heartbeat : heartbeat - now;
    return delta <= threshold;
  }

  bool readFrame(DaedalusFrame& destination, std::chrono::milliseconds timeout) noexcept
  {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    clearError();
    if (metadata_ == nullptr || image_pool_ == nullptr) {
      setError("Daedalus client is not connected");
      return false;
    }

    const auto clamped_timeout = std::max(timeout, std::chrono::milliseconds::zero());
    const auto deadline = std::chrono::steady_clock::now() + clamped_timeout;

    std::optional<Protocol::ImageMeta> image_meta;
    do {
      std::string protocol_error;
      image_meta = consumeLatest(metadata_->image, "image", protocol_error);
      if (!protocol_error.empty()) {
        setError(std::move(protocol_error));
        return false;
      }
      if (image_meta) {
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    } while (true);

    // Direct-copy channels belong to the same published frame.  Copy them
    // before consuming the last pose flag, which is the publisher's signal
    // that it may begin writing the following frame.
    const Protocol::CameraInfo raw_camera_info = metadata_->camera_info;
    const Protocol::ChassisObservation raw_chassis = metadata_->chassis_observation;
    const Protocol::RuntimeState raw_runtime_state = metadata_->runtime_state;

    std::array<Protocol::PoseMeta, Protocol::kSynchronizedPoseCount> raw_poses{};
    bool complete = true;
    std::string protocol_error;
    for (std::size_t index = 0; index < raw_poses.size(); ++index) {
      auto pose = consumeLatest(metadata_->poses[index], "pose", protocol_error);
      if (!pose) {
        complete = false;
      } else {
        raw_poses[index] = *pose;
      }
    }
    if (!protocol_error.empty()) {
      setError(std::move(protocol_error));
      return false;
    }
    if (!complete) {
      setError(
        "incomplete Daedalus frame bundle; only one frame consumer may be connected");
      return false;
    }

    for (const auto& pose : raw_poses) {
      if (pose.frame_sequence != image_meta->sequence ||
          pose.timestamp_ns != image_meta->timestamp_ns) {
        setError("Daedalus image/pose sequence mismatch");
        return false;
      }
    }

    if (image_meta->width == 0 || image_meta->height == 0 ||
        image_meta->width > Protocol::kImageWidth ||
        image_meta->height > Protocol::kImageHeight || image_meta->buffer_id >= 3 ||
        image_meta->format > 2) {
      setError("Daedalus published invalid image metadata");
      return false;
    }

    const std::size_t channels = image_meta->format == 2 ? 1 : 3;
    const std::size_t image_bytes =
      static_cast<std::size_t>(image_meta->width) * image_meta->height * channels;
    const std::size_t image_offset =
      static_cast<std::size_t>(image_meta->buffer_id) * Protocol::kImageSize;
    if (image_offset > image_region_.size() ||
        image_bytes > image_region_.size() - image_offset) {
      setError("Daedalus image points outside the shared image pool");
      return false;
    }

    DaedalusFrame frame;
    frame.sequence = image_meta->sequence;
    frame.timestamp_ns = image_meta->timestamp_ns;
    frame.capture_time = toSteadyTime(frame.timestamp_ns);
    frame.camera_info = convertCameraInfo(raw_camera_info);
    frame.chassis = convertChassis(raw_chassis);
    frame.auto_aim_enabled =
      raw_runtime_state.timestamp_ns != 0 && raw_runtime_state.following != 0;

    for (std::size_t index = 0; index < raw_poses.size(); ++index) {
      frame.poses[index] = {
        .frame_sequence = raw_poses[index].frame_sequence,
        .timestamp_ns = raw_poses[index].timestamp_ns,
        .position = raw_poses[index].position,
        .quaternion = raw_poses[index].quaternion,
      };
    }

    try {
      std::uint8_t* pixels = image_pool_ + image_offset;
      if (image_meta->format == 0) {
        const cv::Mat rgb(
          static_cast<int>(image_meta->height),
          static_cast<int>(image_meta->width),
          CV_8UC3,
          pixels);
        cv::cvtColor(rgb, frame.image_bgr, cv::COLOR_RGB2BGR);
      } else if (image_meta->format == 1) {
        const cv::Mat bgr(
          static_cast<int>(image_meta->height),
          static_cast<int>(image_meta->width),
          CV_8UC3,
          pixels);
        frame.image_bgr = bgr.clone();
      } else {
        const cv::Mat gray(
          static_cast<int>(image_meta->height),
          static_cast<int>(image_meta->width),
          CV_8UC1,
          pixels);
        cv::cvtColor(gray, frame.image_bgr, cv::COLOR_GRAY2BGR);
      }
    } catch (const cv::Exception& error) {
      setError("failed to copy Daedalus image: " + std::string{error.what()});
      return false;
    } catch (const std::exception& error) {
      setError("failed to copy Daedalus image: " + std::string{error.what()});
      return false;
    }

    destination = std::move(frame);
    return true;
  }

  bool sendGimbalCommand(
    float yaw_deg,
    float pitch_deg,
    float distance_m,
    bool fire_advice) noexcept
  {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    clearError();
    if (metadata_ == nullptr) {
      setError("Daedalus client is not connected");
      return false;
    }
    if (!std::isfinite(yaw_deg) || !std::isfinite(pitch_deg) ||
        !std::isfinite(distance_m) || (distance_m < 0.0F && distance_m != -1.0F)) {
      setError("Daedalus gimbal command contains an invalid number or distance");
      return false;
    }

    Protocol::GimbalCommand command;
    command.timestamp_ns = systemNowNs();
    command.yaw_deg = yaw_deg;
    command.pitch_deg = pitch_deg;
    command.distance_m = distance_m;
    command.fire_advice = fire_advice ? 1 : 0;

    std::string protocol_error;
    if (!publishLatest(
          metadata_->gimbal_command, command, "gimbal command", protocol_error)) {
      setError(std::move(protocol_error));
      return false;
    }
    return true;
  }

  DaedalusCameraInfo cameraInfo() const noexcept
  {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    if (metadata_ == nullptr) {
      return {};
    }
    return convertCameraInfo(metadata_->camera_info);
  }

  std::string lastError() const
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
  }

private:
  void disconnectUnlocked() noexcept
  {
    metadata_ = nullptr;
    image_pool_ = nullptr;
    image_region_.reset();
    metadata_region_.reset();
  }

  void clearError() noexcept
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_.clear();
  }

  void setError(std::string error) const noexcept
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = std::move(error);
  }

  DaedalusPaths paths_;
  MappedRegion metadata_region_;
  MappedRegion image_region_;
  Protocol::SharedMetaRegion* metadata_ = nullptr;
  std::uint8_t* image_pool_ = nullptr;
  mutable std::mutex connection_mutex_;
  mutable std::mutex error_mutex_;
  mutable std::string last_error_;
};

bool DaedalusCameraInfo::valid() const noexcept
{
  return timestamp_ns != 0 && width > 0 && height > 0 && std::isfinite(fx) &&
         std::isfinite(fy) && std::isfinite(cx) && std::isfinite(cy) && fx > 0.0 &&
         fy > 0.0;
}

const DaedalusPose& DaedalusFrame::pose(DaedalusPoseKind kind) const noexcept
{
  return poses[static_cast<std::size_t>(kind)];
}

DaedalusClient::DaedalusClient(DaedalusPaths paths)
  : impl_(std::make_unique<Impl>(std::move(paths)))
{
}

DaedalusClient::~DaedalusClient() = default;
DaedalusClient::DaedalusClient(DaedalusClient&&) noexcept = default;
DaedalusClient& DaedalusClient::operator=(DaedalusClient&&) noexcept = default;

bool DaedalusClient::connect() noexcept
{
  return impl_ != nullptr && impl_->connect();
}

void DaedalusClient::disconnect() noexcept
{
  if (impl_ != nullptr) {
    impl_->disconnect();
  }
}

bool DaedalusClient::connected() const noexcept
{
  return impl_ != nullptr && impl_->connected();
}

std::string DaedalusClient::lastError() const
{
  return impl_ != nullptr ? impl_->lastError() : "Daedalus client was moved from";
}

bool DaedalusClient::isSimulatorAlive(std::chrono::milliseconds stale_after) const noexcept
{
  return impl_ != nullptr && impl_->isSimulatorAlive(stale_after);
}

bool DaedalusClient::readFrame(
  DaedalusFrame& frame,
  std::chrono::milliseconds timeout) noexcept
{
  return impl_ != nullptr && impl_->readFrame(frame, timeout);
}

bool DaedalusClient::sendGimbalCommand(
  float yaw_deg,
  float pitch_deg,
  float distance_m,
  bool fire_advice) noexcept
{
  return impl_ != nullptr &&
         impl_->sendGimbalCommand(yaw_deg, pitch_deg, distance_m, fire_advice);
}

DaedalusCameraInfo DaedalusClient::cameraInfo() const noexcept
{
  return impl_ != nullptr ? impl_->cameraInfo() : DaedalusCameraInfo{};
}

}  // namespace L1Sensor
