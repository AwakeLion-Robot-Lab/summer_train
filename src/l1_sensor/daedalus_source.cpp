#include "l1_sensor/daedalus_source.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iterator>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace L1Sensor {
namespace {

// Wire layout adapted from awakening's daedalus_interface (MIT, 2026
// wust-chongshi). Keep these static assertions in sync with
// crates/talos-ipc/src/layout.rs in the simulator.
constexpr std::uint32_t kShmMagic = 0x54414C05;
constexpr std::uint32_t kShmVersion = 2;
constexpr std::uint32_t kImageWidth = 1440;
constexpr std::uint32_t kImageHeight = 1080;
constexpr std::size_t kImageChannels = 3;
constexpr std::size_t kImageSize =
  static_cast<std::size_t>(kImageWidth) * kImageHeight * kImageChannels;
constexpr std::size_t kImagePoolSize = kImageSize * 3;
constexpr std::uint8_t kFlagNew = 0x80;
constexpr std::uint8_t kIndexMask = 0x03;

enum class PoseIndex : std::uint8_t {
  Gimbal = 0,
  Odom = 1,
  Muzzle = 2,
  Camera = 3
};

struct alignas(32) ImageMeta {
  std::uint64_t seq;
  std::uint64_t timestamp_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t buffer_id;
  std::uint8_t format;
  std::uint8_t padding[6];
};
static_assert(sizeof(ImageMeta) == 32);

struct alignas(64) PoseMeta {
  std::uint64_t frame_seq;
  float position[3];
  float quaternion[4];
  std::uint64_t timestamp_ns;
  std::uint8_t padding[16];
};
static_assert(sizeof(PoseMeta) == 64);

struct alignas(32) GimbalCommand {
  std::uint64_t timestamp_ns;
  float yaw_deg;
  float pitch_deg;
  float distance_m;
  std::uint8_t fire_advice;
  std::uint8_t padding[11];
};
static_assert(sizeof(GimbalCommand) == 32);

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
static_assert(sizeof(CameraInfo) == 128);

struct alignas(64) RuntimeState {
  std::uint64_t timestamp_ns;
  std::uint8_t following;
  std::uint8_t padding[55];
};
static_assert(sizeof(RuntimeState) == 64);

struct alignas(64) ImageTripleBuffer {
  alignas(64) std::atomic<std::uint8_t> state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::uint8_t padding[61];
  ImageMeta slots[3];
};
static_assert(sizeof(ImageTripleBuffer) == 192);

struct alignas(64) PoseTripleBuffer {
  alignas(64) std::atomic<std::uint8_t> state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::uint8_t padding[61];
  PoseMeta slots[3];
};
static_assert(sizeof(PoseTripleBuffer) == 256);

struct alignas(64) GimbalTripleBuffer {
  alignas(64) std::atomic<std::uint8_t> state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::uint8_t padding[61];
  GimbalCommand slots[3];
};
static_assert(sizeof(GimbalTripleBuffer) == 192);

struct alignas(64) ShmHeader {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint64_t created_ns;
  std::uint64_t heartbeat_ns;
  std::uint32_t image_width;
  std::uint32_t image_height;
  std::uint8_t padding[32];
};
static_assert(sizeof(ShmHeader) == 64);

struct ShmMetaRegion {
  ShmHeader header;
  ImageTripleBuffer image;
  PoseTripleBuffer poses[5];
  GimbalTripleBuffer gimbal_command;
  CameraInfo camera_info;
  // ChassisObservation (128) + GroundTruthBatch (1664). Neither belongs to
  // the auto-aim input path, but their reserved bytes preserve the ABI.
  std::byte unused_channels[1792];
  RuntimeState runtime_state;
};
static_assert(sizeof(ShmMetaRegion) == 3712);
static_assert(offsetof(ShmMetaRegion, camera_info) == 1728);
static_assert(offsetof(ShmMetaRegion, runtime_state) == 3648);

[[nodiscard]] std::uint64_t systemNowNs() noexcept
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}

[[nodiscard]] std::uint64_t loadSharedU64(
  const std::uint64_t& value) noexcept
{
  return __atomic_load_n(&value, __ATOMIC_ACQUIRE);
}

class MappedFile {
public:
  [[nodiscard]] static std::unique_ptr<MappedFile> open(
    const std::string& path,
    std::size_t required_size,
    std::string& error)
  {
    const int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) {
      error = "cannot open " + path + ": " + std::strerror(errno);
      return nullptr;
    }

    struct stat status {};
    if (::fstat(fd, &status) != 0) {
      error = "cannot stat " + path + ": " + std::strerror(errno);
      ::close(fd);
      return nullptr;
    }
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) < required_size) {
      error = path + " is smaller than the Talos protocol requires";
      ::close(fd);
      return nullptr;
    }

    void* data = ::mmap(
      nullptr, required_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
      error = "cannot mmap " + path + ": " + std::strerror(errno);
      ::close(fd);
      return nullptr;
    }
    return std::unique_ptr<MappedFile>(
      new MappedFile(fd, data, required_size));
  }

  ~MappedFile()
  {
    if (data_ != nullptr && data_ != MAP_FAILED) {
      ::munmap(data_, size_);
    }
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  [[nodiscard]] void* data() const noexcept { return data_; }

private:
  MappedFile(int fd, void* data, std::size_t size)
    : fd_(fd), data_(data), size_(size)
  {
  }

  int fd_{-1};
  void* data_{nullptr};
  std::size_t size_{0};
};

template<typename Buffer, typename Slot>
[[nodiscard]] std::optional<Slot> consume(Buffer& buffer) noexcept
{
  std::uint8_t expected = buffer.state.load(std::memory_order_acquire);
  if ((expected & kFlagNew) == 0) {
    return std::nullopt;
  }

  std::uint8_t ready_index = expected & kIndexMask;
  if (ready_index > 2) {
    return std::nullopt;
  }

  std::uint8_t desired = buffer.read_index;
  if (!buffer.state.compare_exchange_strong(
        expected, desired,
        std::memory_order_acq_rel,
        std::memory_order_acquire)) {
    if ((expected & kFlagNew) == 0) {
      return std::nullopt;
    }
    ready_index = expected & kIndexMask;
    if (ready_index > 2) {
      return std::nullopt;
    }
    desired = buffer.read_index;
    if (!buffer.state.compare_exchange_strong(
          expected, desired,
          std::memory_order_acq_rel,
          std::memory_order_relaxed)) {
      return std::nullopt;
    }
  }

  buffer.read_index = ready_index;
  return buffer.slots[ready_index];
}

void publish(GimbalTripleBuffer& buffer, const GimbalCommand& command) noexcept
{
  buffer.slots[buffer.write_index] = command;
  const std::uint8_t old = buffer.state.exchange(
    static_cast<std::uint8_t>(buffer.write_index | kFlagNew),
    std::memory_order_acq_rel);
  buffer.write_index = old & kIndexMask;
}

[[nodiscard]] bool finiteCameraInfo(const CameraInfo& info) noexcept
{
  if (info.timestamp_ns == 0 || info.width == 0 || info.height == 0 ||
      !std::isfinite(info.fx) || !std::isfinite(info.fy) ||
      !std::isfinite(info.cx) || !std::isfinite(info.cy) ||
      info.fx <= 0.0 || info.fy <= 0.0) {
    return false;
  }
  return std::all_of(
    std::begin(info.distortion), std::end(info.distortion),
    [](double value) { return std::isfinite(value); });
}

}  // namespace

struct DaedalusSource::Impl {
  struct PendingFrame {
    std::optional<ImageMeta> image;
    std::array<std::optional<PoseMeta>, 4> poses;
  };

  Impl(
    DaedalusSourceOptions source_options,
    std::unique_ptr<MappedFile> meta_mapping,
    std::unique_ptr<MappedFile> pool_mapping)
    : options(std::move(source_options)),
      meta_region(std::move(meta_mapping)),
      image_pool_region(std::move(pool_mapping)),
      meta(static_cast<ShmMetaRegion*>(meta_region->data())),
      image_pool(static_cast<std::uint8_t*>(image_pool_region->data())),
      system_anchor_ns(systemNowNs()),
      steady_anchor(std::chrono::steady_clock::now())
  {
  }

  [[nodiscard]] std::chrono::steady_clock::time_point toSteady(
    std::uint64_t timestamp_ns) const noexcept
  {
    constexpr std::uint64_t kMaxDelta =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (timestamp_ns >= system_anchor_ns) {
      const auto delta = std::min(timestamp_ns - system_anchor_ns, kMaxDelta);
      return steady_anchor +
        std::chrono::nanoseconds(static_cast<std::int64_t>(delta));
    }
    const auto delta = std::min(system_anchor_ns - timestamp_ns, kMaxDelta);
    return steady_anchor -
      std::chrono::nanoseconds(static_cast<std::int64_t>(delta));
  }

  [[nodiscard]] bool runtimeFollowing() const noexcept
  {
    const std::uint64_t timestamp_ns =
      loadSharedU64(meta->runtime_state.timestamp_ns);
    if (timestamp_ns == 0 || meta->runtime_state.following == 0) {
      return false;
    }
    const std::uint64_t now_ns = systemNowNs();
    const std::uint64_t timeout_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        options.producer_timeout).count());
    return now_ns >= timestamp_ns && now_ns - timestamp_ns <= timeout_ns;
  }

  void setError(std::string message)
  {
    last_error = std::move(message);
  }

  DaedalusSourceOptions options;
  std::unique_ptr<MappedFile> meta_region;
  std::unique_ptr<MappedFile> image_pool_region;
  ShmMetaRegion* meta{nullptr};
  std::uint8_t* image_pool{nullptr};
  std::uint64_t system_anchor_ns{0};
  std::chrono::steady_clock::time_point steady_anchor{};
  PendingFrame pending;
  std::string last_error;
};

DaedalusSource::DaedalusSource(std::unique_ptr<Impl> impl)
  : impl_(std::move(impl))
{
}

DaedalusSource::~DaedalusSource() = default;

std::unique_ptr<DaedalusSource> DaedalusSource::connect(
  const DaedalusSourceOptions& options,
  std::string* error)
{
  std::string detail;
  auto meta_mapping = MappedFile::open(
    options.meta_path, sizeof(ShmMetaRegion), detail);
  if (!meta_mapping) {
    if (error != nullptr) {
      *error = std::move(detail);
    }
    return nullptr;
  }

  auto* meta = static_cast<ShmMetaRegion*>(meta_mapping->data());
  if (meta->header.magic != kShmMagic) {
    detail = "Talos shared-memory magic mismatch";
  } else if (meta->header.version != kShmVersion) {
    detail = "Talos shared-memory version mismatch: expected " +
      std::to_string(kShmVersion) + ", got " +
      std::to_string(meta->header.version);
  } else if (meta->header.image_width != kImageWidth ||
             meta->header.image_height != kImageHeight) {
    detail = "Talos image layout is not 1440x1080 RGB8";
  }
  if (!detail.empty()) {
    if (error != nullptr) {
      *error = std::move(detail);
    }
    return nullptr;
  }

  auto pool_mapping = MappedFile::open(
    options.image_pool_path, kImagePoolSize, detail);
  if (!pool_mapping) {
    if (error != nullptr) {
      *error = std::move(detail);
    }
    return nullptr;
  }

  auto impl = std::make_unique<Impl>(
    options, std::move(meta_mapping), std::move(pool_mapping));
  return std::unique_ptr<DaedalusSource>(
    new DaedalusSource(std::move(impl)));
}

bool DaedalusSource::producerAlive() const noexcept
{
  const std::uint64_t heartbeat_ns =
    loadSharedU64(impl_->meta->header.heartbeat_ns);
  if (heartbeat_ns == 0) {
    return false;
  }
  const std::uint64_t now_ns = systemNowNs();
  const auto timeout_count = std::chrono::duration_cast<std::chrono::nanoseconds>(
    impl_->options.producer_timeout).count();
  if (timeout_count <= 0 || now_ns < heartbeat_ns) {
    return false;
  }
  return now_ns - heartbeat_ns <= static_cast<std::uint64_t>(timeout_count);
}

std::optional<DaedalusFrame> DaedalusSource::read()
{
  auto& pending = impl_->pending;
  if (!pending.image) {
    pending.image = consume<ImageTripleBuffer, ImageMeta>(impl_->meta->image);
    if (!pending.image) {
      return std::nullopt;
    }
  }

  for (std::size_t index = 0; index < pending.poses.size(); ++index) {
    if (!pending.poses[index]) {
      pending.poses[index] = consume<PoseTripleBuffer, PoseMeta>(
        impl_->meta->poses[index]);
    }
  }
  if (std::any_of(
        pending.poses.begin(), pending.poses.end(),
        [](const auto& pose) { return !pose.has_value(); })) {
    return std::nullopt;
  }

  const ImageMeta image_meta = *pending.image;
  std::array<PoseMeta, 4> pose_meta{};
  for (std::size_t index = 0; index < pose_meta.size(); ++index) {
    pose_meta[index] = *pending.poses[index];
  }
  pending = {};

  const bool synchronized = std::all_of(
    pose_meta.begin(), pose_meta.end(),
    [&image_meta](const PoseMeta& pose) {
      return pose.frame_seq == image_meta.seq &&
             pose.timestamp_ns == image_meta.timestamp_ns;
    });
  if (!synchronized) {
    impl_->setError(
      "discarded a Talos frame because image/pose sequence or timestamp differed");
    return std::nullopt;
  }
  if (image_meta.width != kImageWidth || image_meta.height != kImageHeight ||
      image_meta.buffer_id > 2 || image_meta.format > 2) {
    impl_->setError("discarded a Talos frame with invalid image metadata");
    return std::nullopt;
  }

  const auto convertPose = [this](const PoseMeta& pose)
    -> std::optional<DaedalusPose> {
    DaedalusPose result;
    result.position = Eigen::Vector3d{
      pose.position[0], pose.position[1], pose.position[2]};
    result.orientation = Eigen::Quaterniond{
      pose.quaternion[0], pose.quaternion[1],
      pose.quaternion[2], pose.quaternion[3]};
    if (!result.position.allFinite() ||
        !result.orientation.coeffs().allFinite() ||
        result.orientation.squaredNorm() <= 1e-12) {
      return std::nullopt;
    }
    result.orientation.normalize();
    result.frame_seq = pose.frame_seq;
    result.timestamp_ns = pose.timestamp_ns;
    result.timestamp = impl_->toSteady(pose.timestamp_ns);
    return result;
  };

  const auto gimbal = convertPose(pose_meta[static_cast<std::size_t>(PoseIndex::Gimbal)]);
  const auto odom = convertPose(pose_meta[static_cast<std::size_t>(PoseIndex::Odom)]);
  const auto muzzle = convertPose(pose_meta[static_cast<std::size_t>(PoseIndex::Muzzle)]);
  const auto camera = convertPose(pose_meta[static_cast<std::size_t>(PoseIndex::Camera)]);
  if (!gimbal || !odom || !muzzle || !camera) {
    impl_->setError("discarded a Talos frame containing an invalid pose");
    return std::nullopt;
  }

  const std::size_t offset =
    static_cast<std::size_t>(image_meta.buffer_id) * kImageSize;
  cv::Mat mapped;
  if (image_meta.format == 2) {
    mapped = cv::Mat(
      static_cast<int>(image_meta.height),
      static_cast<int>(image_meta.width), CV_8UC1,
      impl_->image_pool + offset);
  } else {
    mapped = cv::Mat(
      static_cast<int>(image_meta.height),
      static_cast<int>(image_meta.width), CV_8UC3,
      impl_->image_pool + offset);
  }

  DaedalusFrame frame;
  try {
    if (image_meta.format == 0) {
      cv::cvtColor(mapped, frame.bgr_image, cv::COLOR_RGB2BGR);
    } else if (image_meta.format == 1) {
      frame.bgr_image = mapped.clone();
    } else {
      cv::cvtColor(mapped, frame.bgr_image, cv::COLOR_GRAY2BGR);
    }
  } catch (const cv::Exception& exception) {
    impl_->setError(
      std::string("failed to copy a Talos image: ") + exception.what());
    return std::nullopt;
  }

  frame.frame_seq = image_meta.seq;
  frame.timestamp_ns = image_meta.timestamp_ns;
  frame.timestamp = impl_->toSteady(image_meta.timestamp_ns);
  frame.gimbal = *gimbal;
  frame.odom = *odom;
  frame.muzzle = *muzzle;
  frame.camera = *camera;
  frame.following = impl_->runtimeFollowing();
  return frame;
}

std::optional<CameraCalibration> DaedalusSource::calibration(
  const DaedalusFrame& frame,
  std::string* error) const
{
  const CameraInfo info = impl_->meta->camera_info;
  if (!finiteCameraInfo(info)) {
    if (error != nullptr) {
      *error = "Talos CameraInfo is missing or invalid";
    }
    return std::nullopt;
  }
  if (info.width != static_cast<std::uint32_t>(frame.bgr_image.cols) ||
      info.height != static_cast<std::uint32_t>(frame.bgr_image.rows)) {
    if (error != nullptr) {
      *error = "Talos CameraInfo resolution does not match the image";
    }
    return std::nullopt;
  }

  CameraCalibration result;
  result.image_size = {
    static_cast<int>(info.width), static_cast<int>(info.height)};
  result.camera_matrix = cv::Mat::eye(3, 3, CV_64FC1);
  result.camera_matrix.at<double>(0, 0) = info.fx;
  result.camera_matrix.at<double>(1, 1) = info.fy;
  result.camera_matrix.at<double>(0, 2) = info.cx;
  result.camera_matrix.at<double>(1, 2) = info.cy;
  result.distortion_coefficients = cv::Mat(1, 5, CV_64FC1);
  std::copy(
    std::begin(info.distortion), std::end(info.distortion),
    result.distortion_coefficients.ptr<double>());

  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.linear() <<
     0.0,  0.0, 1.0,
    -1.0,  0.0, 0.0,
     0.0, -1.0, 0.0;
  // Camera and muzzle translations are both expressed in the Talos gimbal /
  // barrel axes, so their difference is the camera origin in the muzzle frame.
  transform.translation() = frame.camera.position - frame.muzzle.position;
  if (!transform.translation().allFinite()) {
    if (error != nullptr) {
      *error = "Talos camera/muzzle extrinsics are not finite";
    }
    return std::nullopt;
  }
  result.T_barrel_camera = transform;
  return result;
}

bool DaedalusSource::sendGimbalCommand(
  double yaw_rad,
  double pitch_rad,
  double distance_m,
  bool fire_advice) noexcept
{
  if (!std::isfinite(yaw_rad) || !std::isfinite(pitch_rad) ||
      !std::isfinite(distance_m) || distance_m < 0.0) {
    return false;
  }

  constexpr double kRadToDeg = 180.0 / std::numbers::pi;
  const double yaw_deg = yaw_rad * kRadToDeg;
  const double pitch_deg = pitch_rad * kRadToDeg;
  if (std::abs(yaw_deg) > std::numeric_limits<float>::max() ||
      std::abs(pitch_deg) > std::numeric_limits<float>::max() ||
      distance_m > std::numeric_limits<float>::max()) {
    return false;
  }

  GimbalCommand command{};
  command.timestamp_ns = systemNowNs();
  command.yaw_deg = static_cast<float>(yaw_deg);
  command.pitch_deg = static_cast<float>(pitch_deg);
  command.distance_m = static_cast<float>(distance_m);
  command.fire_advice = fire_advice ? 1U : 0U;
  publish(impl_->meta->gimbal_command, command);
  return true;
}

void DaedalusSource::sendHold() noexcept
{
  GimbalCommand command{};
  command.timestamp_ns = systemNowNs();
  command.distance_m = -1.0F;
  publish(impl_->meta->gimbal_command, command);
}

std::string DaedalusSource::takeLastError()
{
  std::string result = std::move(impl_->last_error);
  impl_->last_error.clear();
  return result;
}

}  // namespace L1Sensor
